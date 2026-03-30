"""Dynamic Gaussian Clustering via Faiss GPU KMeans.

Clusters dynamic (high-dynamic-probability) Gaussians using a weighted
combination of:
  1. 3D position (xyz)
  2. SH 0th-order coefficients (DC color ~ RGB)
  3. Historical mean deformation displacement

Produces per-Gaussian cluster labels and quality metrics (silhouette score,
inter/intra-cluster distance ratio) for evaluation.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Feature extraction & normalisation
# ---------------------------------------------------------------------------

def _z_normalise(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column z-score normalisation.  Returns (normed, mean, std)."""
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + 1e-8
    return (x - mu) / sigma, mu, sigma


def _build_feature_matrix(
    xyz: np.ndarray,
    sh_dc: np.ndarray,
    mean_deform: np.ndarray,
    w_xyz: float,
    w_color: float,
    w_motion: float,
) -> np.ndarray:
    """Build the weighted & normalised feature matrix for KMeans.

    Each sub-feature is z-normalised independently, then multiplied by its
    weight so that the Euclidean distance used by KMeans respects the intended
    relative importance.

    Returns:
        features – (N, D) float32 array ready for Faiss.
    """
    parts = []
    if w_xyz > 0:
        n_xyz, _, _ = _z_normalise(xyz)
        parts.append(n_xyz * w_xyz)
    if w_color > 0:
        n_color, _, _ = _z_normalise(sh_dc)
        parts.append(n_color * w_color)
    if w_motion > 0:
        n_motion, _, _ = _z_normalise(mean_deform)
        parts.append(n_motion * w_motion)
    features = np.concatenate(parts, axis=1).astype(np.float32)
    return np.ascontiguousarray(features)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def _intra_inter_ratio(features: np.ndarray, labels: np.ndarray, k: int) -> Tuple[float, float, float]:
    """Compute intra-cluster compactness, inter-cluster separation and their ratio.

    Returns (intra, inter, ratio=intra/inter).  Lower ratio ⇒ better clustering.
    """
    centroids = np.zeros((k, features.shape[1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(axis=0)
            counts[c] = mask.sum()

    # intra: mean of per-point distance to own centroid
    intra = 0.0
    for c in range(k):
        mask = labels == c
        if counts[c] > 0:
            diff = features[mask] - centroids[c]
            intra += np.sqrt((diff ** 2).sum(axis=1)).sum()
    intra /= max(labels.shape[0], 1)

    # inter: mean pairwise centroid distance
    inter = 0.0
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            if counts[i] > 0 and counts[j] > 0:
                inter += np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
                n_pairs += 1
    inter /= max(n_pairs, 1)

    ratio = intra / max(inter, 1e-8)
    return float(intra), float(inter), float(ratio)


def _silhouette_sampled(features: np.ndarray, labels: np.ndarray, k: int,
                        max_samples: int = 10000) -> float:
    """Approximate silhouette score via random sampling for efficiency."""
    n = features.shape[0]
    if n <= 1 or k <= 1:
        return 0.0
    # Sample
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        n = max_samples

    # Pre-compute centroids for speed (simplified silhouette using centroid distances)
    centroids = np.zeros((k, features.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(axis=0)

    sil_vals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own_c = labels[i]
        a_i = np.sqrt(((features[i] - centroids[own_c]) ** 2).sum())
        # nearest other centroid
        b_i = float("inf")
        for c in range(k):
            if c != own_c:
                d = np.sqrt(((features[i] - centroids[c]) ** 2).sum())
                if d < b_i:
                    b_i = d
        if b_i == float("inf"):
            b_i = 0.0
        sil_vals[i] = (b_i - a_i) / max(a_i, b_i, 1e-8)

    return float(sil_vals.mean())


# ---------------------------------------------------------------------------
# Main clustering entry point
# ---------------------------------------------------------------------------

def cluster_dynamic_gaussians(
    gaussians,
    dynamic_thresh: float = 0.5,
    n_clusters: int = 8,
    w_xyz: float = 1.0,
    w_color: float = 0.5,
    w_motion: float = 1.0,
    temperature: float = 1.0,
    tb_writer=None,
    iteration: int = 0,
    dynamic_score_percentile: float = 80.0,
) -> Dict[str, object]:
    """Cluster dynamic Gaussians and return results.

    Args:
        gaussians: GaussianModel instance.
        dynamic_thresh: Probability threshold for selecting dynamic Gaussians (legacy).
        n_clusters: Number of KMeans clusters.
        w_xyz / w_color / w_motion: Feature weights.
        temperature: Sigmoid temperature for dynamic prob.
        tb_writer: Optional TensorBoard writer.
        iteration: Current training iteration (for logging).
        save_path: If provided, save cluster labels as .npz.
        dynamic_score_percentile: Percentile threshold for dynamic score (0-100).
            Default 80 means select Gaussians with dynamic score in top 20%.

    Returns:
        Dict with keys:
          - "labels": (N_total,) int32 tensor, -1 for static, ≥0 for cluster id
          - "dynamic_mask": (N_total,) bool tensor
          - "n_dynamic": int
          - "metrics": dict of quality metrics
    """
    t_start = time.perf_counter()
    N = gaussians.get_xyz.shape[0]

    # ── 1. Select dynamic Gaussians ──
    # Use dynamic score (from iter 15000) if available, otherwise fall back to dynamic probability
    with torch.no_grad():
        # Use dynamic score mechanism: compute score and compute percentile-based threshold
        dyn_score = gaussians.compute_dynamic_score()  # (N,), range [0, 1]
        
        # Debug logging
        print(f"[DEBUG] dyn_score: min={dyn_score.min().item():.6f}, max={dyn_score.max().item():.6f}, mean={dyn_score.mean().item():.6f}")
        print(f"[DEBUG] N_gaussians={N}, percentile={dynamic_score_percentile}")
            
        # Compute threshold based on percentile: get the value at (percentile) position
        # e.g., percentile=80 means select Gaussians with score >= 80th percentile (top 20%)
        sorted_scores, _ = torch.sort(dyn_score)
        
        # For percentile=80, we want to select scores >= 80th percentile value
        # So we find the index at percentile position and use that value as threshold
        percentile_idx = int(dynamic_score_percentile / 100.0 * N)
        percentile_idx = max(0, min(percentile_idx, N - 1))
        score_thresh = sorted_scores[percentile_idx].item()
        
        print(f"[DEBUG] percentile_idx={percentile_idx}, score_thresh={score_thresh:.6f}")
        
        # Select Gaussians with score >= threshold (top (100-percentile)%)
        dynamic_mask = dyn_score >= score_thresh  # boolean (N,)
        n_dynamic = int(dynamic_mask.sum().item())
        
        print(f"[DEBUG] Selected {n_dynamic} dynamic Gaussians out of {N}")
            
        _msg = (f"[CLUSTER] Using dynamic score mechanism: "
                f"percentile={dynamic_score_percentile}, "
                f"score_threshold={score_thresh:.4f}, "
                f"selected={n_dynamic}/{N}")
        logger.info(_msg)
        print(_msg)
            
        # Log score distribution for debugging
        if tb_writer is not None:
            tb_writer.add_scalar('cluster/dynamic_score_thresh', score_thresh, iteration)
            tb_writer.add_scalar('cluster/n_dynamic_gaussians', n_dynamic, iteration)
            tb_writer.add_histogram('cluster/dynamic_scores', dyn_score, iteration)

    _msg = f"[CLUSTER] Selecting dynamic Gaussians: {n_dynamic}/{N}"
    logger.info(_msg)
    print(_msg)

    if n_dynamic < n_clusters:
        _warn = (f"[CLUSTER] Too few dynamic Gaussians ({n_dynamic}) for {n_clusters} clusters. "
                 "Skipping clustering.")
        logger.warning(_warn)
        print(_warn)
        labels_full = torch.full((N,), -1, dtype=torch.int32, device="cuda")
        return {"labels": labels_full, "dynamic_mask": dynamic_mask,
                "n_dynamic": n_dynamic, "metrics": {}}

    # ── 2. Gather features ──
    with torch.no_grad():
        xyz_np = gaussians.get_xyz[dynamic_mask].detach().cpu().numpy()  # (M, 3)
        # SH DC: _features_dc is (N, 1, 3) → squeeze to (N, 3)
        sh_dc_np = gaussians._features_dc[dynamic_mask].detach().squeeze(1).cpu().numpy()  # (M, 3)
        # Mean deformation
        if hasattr(gaussians, 'get_mean_deform'):
            mean_def_np = gaussians.get_mean_deform()[dynamic_mask].detach().cpu().numpy()  # (M, 3)
        else:
            mean_def_np = np.zeros_like(xyz_np)
            logger.warning("[CLUSTER] No deform history available, using zero motion features")

    _msg = (f"[CLUSTER] Building feature matrix: "
            f"xyz(w={w_xyz}), color(w={w_color}), motion(w={w_motion})")
    logger.info(_msg)
    print(_msg)

    features = _build_feature_matrix(xyz_np, sh_dc_np, mean_def_np,
                                     w_xyz, w_color, w_motion)
    feat_dim = features.shape[1]
    _msg = f"[CLUSTER] Feature matrix: {features.shape[0]} points × {feat_dim} dims"
    logger.info(_msg)
    print(_msg)

    # ── 3. Faiss GPU KMeans ──
    _msg = f"[CLUSTER] Running Faiss GPU KMeans (k={n_clusters}) ..."
    logger.info(_msg)
    print(_msg)

    try:
        import faiss
        # Use single GPU (device 0)
        res = faiss.StandardGpuResources()
        kmeans = faiss.Clustering(feat_dim, n_clusters)
        kmeans.niter = 30
        kmeans.verbose = True
        kmeans.seed = 42

        # Build a flat L2 index on GPU
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0
        gpu_index = faiss.GpuIndexFlatL2(res, feat_dim, cfg)

        kmeans.train(features, gpu_index)

        # Assign labels
        _, labels_np = gpu_index.search(features, 1)
        labels_np = labels_np.squeeze(-1).astype(np.int32)  # (M,)
    except ImportError:
        _err = "[CLUSTER] faiss-gpu not installed. Install with: pip install faiss-gpu"
        logger.error(_err)
        print(_err)
        labels_full = torch.full((N,), -1, dtype=torch.int32, device="cuda")
        return {"labels": labels_full, "dynamic_mask": dynamic_mask,
                "n_dynamic": n_dynamic, "metrics": {}}

    t_kmeans = time.perf_counter()
    _msg = f"[CLUSTER] KMeans done in {t_kmeans - t_start:.2f}s"
    logger.info(_msg)
    print(_msg)

    # ── 4. Quality metrics ──
    _msg = "[CLUSTER] Computing quality metrics ..."
    logger.info(_msg)
    print(_msg)

    intra, inter, ratio = _intra_inter_ratio(features, labels_np, n_clusters)
    sil = _silhouette_sampled(features, labels_np, n_clusters, max_samples=10000)

    # Per-cluster stats
    cluster_counts = np.bincount(labels_np, minlength=n_clusters)

    metrics = {
        "silhouette": sil,
        "intra": intra,
        "inter": inter,
        "intra_inter_ratio": ratio,
        "cluster_counts": cluster_counts.tolist(),
    }

    _msg = (f"[CLUSTER] Quality:  silhouette={sil:.4f}  "
            f"intra={intra:.4f}  inter={inter:.4f}  ratio={ratio:.4f}")
    logger.info(_msg)
    print(_msg)
    for c_id in range(n_clusters):
        _msg = f"[CLUSTER]   cluster {c_id}: {cluster_counts[c_id]} Gaussians"
        logger.info(_msg)
        print(_msg)

    # ── 5. Assemble full-size label tensor ──
    labels_full = torch.full((N,), -1, dtype=torch.int32, device="cuda")
    dynamic_indices = torch.where(dynamic_mask)[0]
    labels_torch = torch.from_numpy(labels_np).to(device="cuda", dtype=torch.int32)
    labels_full[dynamic_indices] = labels_torch

    # ── 6. TensorBoard logging ──
    if tb_writer is not None:
        tb_writer.add_scalar('cluster/n_dynamic', n_dynamic, iteration)
        tb_writer.add_scalar('cluster/silhouette', sil, iteration)
        tb_writer.add_scalar('cluster/intra', intra, iteration)
        tb_writer.add_scalar('cluster/inter', inter, iteration)
        tb_writer.add_scalar('cluster/intra_inter_ratio', ratio, iteration)
        for c_id in range(n_clusters):
            tb_writer.add_scalar(f'cluster/count_{c_id}', int(cluster_counts[c_id]), iteration)

    t_end = time.perf_counter()
    _msg = f"[CLUSTER] Total clustering time: {t_end - t_start:.2f}s"
    logger.info(_msg)
    print(_msg)

    return {
        "labels": labels_full,
        "dynamic_mask": dynamic_mask,
        "n_dynamic": n_dynamic,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Debug pseudo-color render
# ---------------------------------------------------------------------------

# 8 high-contrast colors (Tab10-inspired) for cluster visualisation
_CLUSTER_PALETTE = torch.tensor([
    [1.00, 0.20, 0.20],   # red
    [0.20, 0.60, 1.00],   # blue
    [0.17, 0.80, 0.27],   # green
    [1.00, 0.60, 0.00],   # orange
    [0.60, 0.30, 0.90],   # purple
    [0.00, 0.80, 0.80],   # cyan
    [0.96, 0.80, 0.00],   # yellow
    [0.96, 0.40, 0.70],   # pink
], dtype=torch.float32)  # (8, 3)

_STATIC_COLOR = torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32)  # dark grey


def render_cluster_pseudocolor(
    gaussians,
    labels: torch.Tensor,
    viewpoint_cam,
    deform,
    pipe,
    bg_color: torch.Tensor,
    mult: float,
    is_6dof: bool = False,
    save_path: Optional[str] = None,
    tb_writer=None,
    iteration: int = 0,
) -> torch.Tensor:
    """Render a pseudo-color image where each dynamic cluster has a distinct colour.

    Static Gaussians (label == -1) are rendered dark grey.
    Dynamic Gaussians are coloured by their cluster label (0..K-1).

    The function temporarily hijacks the rendering pipeline by calling the
    rasterizer directly with ``colors_precomp`` instead of SH features,
    leaving the original GaussianModel untouched.

    Args:
        gaussians: GaussianModel with clustering labels.
        labels: (N,) int32 tensor, -1 = static, >=0 = cluster id.
        viewpoint_cam: Camera object (cam00, fid=0).
        deform: Deformation network.
        pipe: PipelineParams.
        bg_color: Background colour tensor (3,) on GPU.
        mult: FastGS mult parameter.
        is_6dof: Whether this is a 6-DoF scene.
        save_path: If provided, save the rendered image as PNG.
        tb_writer: Optional TensorBoard writer.
        iteration: Current training iteration.

    Returns:
        rendered_image – (3, H, W) float tensor in [0, 1].
    """
    import math
    import os
    from diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from utils.rigid_utils import from_homogenous, to_homogenous

    N = gaussians.get_xyz.shape[0]

    # ── 1. Build per-Gaussian pseudo colour ──
    palette = _CLUSTER_PALETTE.to(device="cuda")  # (8, 3)
    static_c = _STATIC_COLOR.to(device="cuda")    # (3,)

    colors = static_c.unsqueeze(0).expand(N, 3).clone()  # default: dark grey
    for c_id in range(palette.shape[0]):
        mask = labels == c_id
        if mask.any():
            colors[mask] = palette[c_id]

    # ── 2. Compute deformation at viewpoint time ──
    with torch.no_grad():
        fid = viewpoint_cam.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

        if is_6dof and torch.is_tensor(d_xyz):
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(xyz).unsqueeze(-1)).squeeze(-1))
        elif torch.is_tensor(d_xyz):
            means3D = xyz + d_xyz
        else:
            means3D = xyz

        scales = gaussians.get_scaling + (d_scaling if torch.is_tensor(d_scaling) else 0)
        rotations = gaussians.get_rotation + (d_rotation if torch.is_tensor(d_rotation) else 0)
        opacity = gaussians.get_opacity

    # ── 3. Set up rasterizer ──
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
    H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)

    screenspace_points = torch.zeros(N, 4, dtype=torch.float32, device="cuda")
    metric_map = torch.zeros(H * W, dtype=torch.int, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=0,  # irrelevant — using colors_precomp
        campos=viewpoint_cam.camera_center,
        mult=mult,
        prefiltered=False,
        debug=False,
        get_flag=None,
        metric_map=metric_map,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ── 4. Rasterize with precomputed colors ──
    with torch.no_grad():
        rendered_image, _, _, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            opacities=opacity,
            colors_precomp=colors,  # (N, 3) — bypass SH entirely
            scales=scales,
            rotations=rotations,
        )

    rendered_image = rendered_image.clamp(0.0, 1.0)

    # ── 5. Save / log ──
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import torchvision
        torchvision.utils.save_image(rendered_image, save_path)
        _msg = f"[CLUSTER-VIS] Pseudo-color render saved to {save_path}"
        logger.info(_msg)
        print(_msg)

    if tb_writer is not None:
        tb_writer.add_image("cluster/pseudocolor", rendered_image, iteration)

    return rendered_image


# ---------------------------------------------------------------------------
# Capacity allocation by dynamic score
# ---------------------------------------------------------------------------

# 8 high-contrast colors (Tab10-inspired) for cluster visualisation
_CLUSTER_PALETTE = torch.tensor([
    [1.00, 0.20, 0.20],   # red
    [0.20, 0.60, 1.00],   # blue
    [0.17, 0.80, 0.27],   # green
    [1.00, 0.60, 0.00],   # orange
    [0.60, 0.30, 0.90],   # purple
    [0.00, 0.80, 0.80],   # cyan
    [0.96, 0.80, 0.00],   # yellow
    [0.96, 0.40, 0.70],   # pink
], dtype=torch.float32)  # (8, 3)

_STATIC_COLOR = torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32)  # dark grey


def allocate_capacity_by_score(
    cluster_mean_scores: list[float],
    n_clusters: int,
    capacity_tier_configs: Optional[Dict] = None,
    min_spatial_res: Tuple[int, ...] = (64, 96),
    max_spatial_res: Tuple[int, ...] = (64, 128, 192),
    min_time_res: Tuple[int, ...] = (64, 96),
    max_time_res: Tuple[int, ...] = (64, 128, 192),
    min_mlp_hidden: int = 48,
    max_mlp_hidden: int = 96,
    min_feat_dim: int = 8,
    max_feat_dim: int = 12,
    strategy: str = "tiered",
    tier_boundaries: Optional[List[float]] = None,
) -> List[Dict]:
    """Allocate capacity for each cluster based on mean dynamic scores.
    
    Args:
        cluster_mean_scores: List of average dynamic score for each cluster.
        n_clusters: Number of clusters.
        capacity_tier_configs: Predefined tier configurations (from JSON).
        min_spatial_res: Minimum spatial resolutions (tuple of levels).
        max_spatial_res: Maximum spatial resolutions (tuple of levels).
        min_time_res: Minimum temporal resolutions.
        max_time_res: Maximum temporal resolutions.
        min_mlp_hidden: Minimum MLP hidden dimension.
        max_mlp_hidden: Maximum MLP hidden dimension.
        min_feat_dim: Minimum feature dimension.
        max_feat_dim: Maximum feature dimension.
        strategy: Allocation strategy ("tiered" or "linear").
        tier_boundaries: Boundaries for tiered strategy (e.g., [0.33, 0.67]).
    
    Returns:
        student_configs: List of dicts, each containing capacity config for one cluster.
    """
    assert len(cluster_mean_scores) == n_clusters, "Score count must match cluster count"
    
    if strategy == "tiered":
        return _allocate_tiered(
            cluster_mean_scores, n_clusters, capacity_tier_configs,
            min_spatial_res, max_spatial_res, min_time_res, max_time_res,
            min_mlp_hidden, max_mlp_hidden, min_feat_dim, max_feat_dim,
            tier_boundaries
        )
    elif strategy == "linear":
        return _allocate_linear(
            cluster_mean_scores, n_clusters,
            min_spatial_res, max_spatial_res, min_time_res, max_time_res,
            min_mlp_hidden, max_mlp_hidden, min_feat_dim, max_feat_dim
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'tiered' or 'linear'.")


def _allocate_tiered(
    cluster_mean_scores: list[float],
    n_clusters: int,
    capacity_tier_configs: Optional[Dict],
    min_spatial_res: Tuple[int, ...],
    max_spatial_res: Tuple[int, ...],
    min_time_res: Tuple[int, ...],
    max_time_res: Tuple[int, ...],
    min_mlp_hidden: int,
    max_mlp_hidden: int,
    min_feat_dim: int,
    max_feat_dim: int,
    tier_boundaries: Optional[List[float]],
) -> List[Dict]:
    """3-tier allocation: high/medium/low based on score ranking."""
    if tier_boundaries is None:
        tier_boundaries = [0.33, 0.67]
    
    # Get sorted indices (descending by score)
    sorted_indices = sorted(range(n_clusters), key=lambda i: cluster_mean_scores[i], reverse=True)
    
    # Determine tier boundaries in terms of cluster count
    n_high = max(1, int(n_clusters * tier_boundaries[0]))
    n_medium = max(1, int(n_clusters * tier_boundaries[1]) - n_high)
    # Low gets the rest
    
    # Use predefined configs if available
    if capacity_tier_configs is not None and "tiers" in capacity_tier_configs:
        tiers = capacity_tier_configs["tiers"]
        high_config = {
            "spatial_resolutions": tiers["high"]["spatial_resolutions"],
            "time_resolutions": tiers["high"]["time_resolutions"],
            "mlp_hidden_dim": tiers["high"]["mlp_hidden_dim"],
            "feat_dim": tiers["high"]["feat_dim"],
        }
        medium_config = {
            "spatial_resolutions": tiers["medium"]["spatial_resolutions"],
            "time_resolutions": tiers["medium"]["time_resolutions"],
            "mlp_hidden_dim": tiers["medium"]["mlp_hidden_dim"],
            "feat_dim": tiers["medium"]["feat_dim"],
        }
        low_config = {
            "spatial_resolutions": tiers["low"]["spatial_resolutions"],
            "time_resolutions": tiers["low"]["time_resolutions"],
            "mlp_hidden_dim": tiers["low"]["mlp_hidden_dim"],
            "feat_dim": tiers["low"]["feat_dim"],
        }
    else:
        # Fallback: construct configs from min/max parameters
        high_config = {
            "spatial_resolutions": list(max_spatial_res),
            "time_resolutions": list(max_time_res),
            "mlp_hidden_dim": max_mlp_hidden,
            "feat_dim": max_feat_dim,
        }
        medium_config = {
            "spatial_resolutions": list(min_spatial_res) + [(min_spatial_res[-1] + max_spatial_res[-1]) // 2],
            "time_resolutions": list(min_time_res) + [(min_time_res[-1] + max_time_res[-1]) // 2],
            "mlp_hidden_dim": (min_mlp_hidden + max_mlp_hidden) // 2,
            "feat_dim": (min_feat_dim + max_feat_dim) // 2,
        }
        low_config = {
            "spatial_resolutions": list(min_spatial_res),
            "time_resolutions": list(min_time_res),
            "mlp_hidden_dim": min_mlp_hidden,
            "feat_dim": min_feat_dim,
        }
    
    # Allocate configs
    student_configs = [None] * n_clusters
    for rank, cluster_idx in enumerate(sorted_indices):
        if rank < n_high:
            student_configs[cluster_idx] = high_config
        elif rank < n_high + n_medium:
            student_configs[cluster_idx] = medium_config
        else:
            student_configs[cluster_idx] = low_config
    
    return student_configs


def _allocate_linear(
    cluster_mean_scores: list[float],
    n_clusters: int,
    min_spatial_res: Tuple[int, ...],
    max_spatial_res: Tuple[int, ...],
    min_time_res: Tuple[int, ...],
    max_time_res: Tuple[int, ...],
    min_mlp_hidden: int,
    max_mlp_hidden: int,
    min_feat_dim: int,
    max_feat_dim: int,
) -> List[Dict]:
    """Linear interpolation: capacity scales continuously with score."""
    scores = np.array(cluster_mean_scores)
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score + 1e-8
    
    # Normalize scores to [0, 1]
    norm_scores = (scores - min_score) / score_range
    
    student_configs = []
    for i in range(n_clusters):
        alpha = norm_scores[i]  # Interpolation factor
        
        # Interpolate spatial resolutions (element-wise)
        spatial_res = tuple(
            int(min_r + alpha * (max_r - min_r))
            for min_r, max_r in zip(min_spatial_res, max_spatial_res)
        )
        
        # Interpolate time resolutions
        time_res = tuple(
            int(min_r + alpha * (max_r - min_r))
            for min_r, max_r in zip(min_time_res, max_time_res)
        )
        
        # Interpolate MLP hidden dim and feat_dim
        mlp_hidden = int(min_mlp_hidden + alpha * (max_mlp_hidden - min_mlp_hidden))
        feat_dim = int(min_feat_dim + alpha * (max_feat_dim - min_feat_dim))
        
        student_configs.append({
            "spatial_resolutions": spatial_res,
            "time_resolutions": time_res,
            "mlp_hidden_dim": mlp_hidden,
            "feat_dim": feat_dim,
        })
    
    return student_configs


def compute_cluster_mean_scores(
    gaussians,
    cluster_labels: torch.Tensor,
    n_clusters: int,
) -> List[float]:
    """Compute mean dynamic score for each cluster.
    
    Args:
        gaussians: GaussianModel with dynamic scores.
        cluster_labels: (N,) int32 tensor with cluster assignments (-1 for static).
        n_clusters: Number of clusters.
    
    Returns:
        cluster_mean_scores: List of mean scores per cluster.
    """
    with torch.no_grad():
        dyn_score = gaussians.compute_dynamic_score()  # (N,), range [0, 1]
        
        cluster_mean_scores = []
        for k in range(n_clusters):
            mask = (cluster_labels == k)
            if mask.sum() > 0:
                mean_score = dyn_score[mask].mean().item()
            else:
                mean_score = 0.0
            cluster_mean_scores.append(mean_score)
    
    return cluster_mean_scores
