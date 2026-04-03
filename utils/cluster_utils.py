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
        # Deformation variance (replaces mean deformation for better high-freq separation)
        if hasattr(gaussians, 'get_deform_var'):
            deform_var_np = gaussians.get_deform_var()[dynamic_mask].detach().cpu().numpy()  # (M, 3)
        else:
            deform_var_np = np.zeros_like(xyz_np)
            logger.warning("[CLUSTER] No deform variance available, using zero motion features")

    _msg = (f"[CLUSTER] Building feature matrix: "
            f"xyz(w={w_xyz}), color(w={w_color}), motion(w={w_motion})")
    logger.info(_msg)
    print(_msg)

    features = _build_feature_matrix(xyz_np, sh_dc_np, deform_var_np,
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
        return _allocate_tiered(cluster_mean_scores, n_clusters, capacity_tier_configs,tier_boundaries)
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
    if capacity_tier_configs is not None:
        tiers = capacity_tier_configs["tiered"]["tiers"]
        high_config = {
            "tier": "high",
            "spatial_resolutions": tiers["high"]["spatial_resolutions"],
            "time_resolutions": tiers["high"]["time_resolutions"],
            "mlp_hidden_dim": tiers["high"]["mlp_hidden_dim"],
            "mlp_layer_num": tiers["high"].get("mlp_layer_num", 2),
            "feat_dim": tiers["high"]["feat_dim"],
        }
        medium_config = {
            "tier": "medium",
            "spatial_resolutions": tiers["medium"]["spatial_resolutions"],
            "time_resolutions": tiers["medium"]["time_resolutions"],
            "mlp_hidden_dim": tiers["medium"]["mlp_hidden_dim"],
            "mlp_layer_num": tiers["medium"].get("mlp_layer_num", 2),
            "feat_dim": tiers["medium"]["feat_dim"],
        }
        low_config = {
            "tier": "low",
            "spatial_resolutions": tiers["low"]["spatial_resolutions"],
            "time_resolutions": tiers["low"]["time_resolutions"],
            "mlp_hidden_dim": tiers["low"]["mlp_hidden_dim"],
            "mlp_layer_num": tiers["low"].get("mlp_layer_num", 2),
            "feat_dim": tiers["low"]["feat_dim"],
        }
        print("[INFO] Load capacity tier configs successfully!")
    else:
        raise ValueError(f"[ERROR] capacity tier configs not found, please check tier config path")

    # Allocate configs
    student_configs = [None] * n_clusters
    for rank, cluster_idx in enumerate(sorted_indices):
        if rank < n_high:
            student_configs[cluster_idx] = high_config.copy()
        elif rank < n_high + n_medium:
            student_configs[cluster_idx] = medium_config.copy()
        else:
            student_configs[cluster_idx] = low_config.copy()
    
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
    """Linear interpolation: capacity scales continuously with score.
    
    Each config includes a 'tier' field inferred from the capacity ranking.
    """
    scores = np.array(cluster_mean_scores)
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score + 1e-8
    
    # Normalize scores to [0, 1]
    norm_scores = (scores - min_score) / score_range
    
    # Compute capacity ranking for tier assignment
    sorted_indices = sorted(range(n_clusters), key=lambda i: norm_scores[i], reverse=True)
    n_tier = max(1, n_clusters // 3)
    
    # Assign tiers based on ranking
    tier_labels = [""] * n_clusters
    for rank, cluster_idx in enumerate(sorted_indices):
        if rank < n_tier:
            tier_labels[cluster_idx] = "high"
        elif rank < 2 * n_tier:
            tier_labels[cluster_idx] = "medium"
        else:
            tier_labels[cluster_idx] = "low"
    
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
            "tier": tier_labels[i],
            "spatial_resolutions": spatial_res,
            "time_resolutions": time_res,
            "mlp_hidden_dim": mlp_hidden,
            "mlp_layer_num": 2,  # Default to 2 layers
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


# ---------------------------------------------------------------------------
# Capacity allocation visualization
# ---------------------------------------------------------------------------

# Capacity tier colors (red=high, yellow=medium, blue=low)
_CAPACITY_TIER_COLORS = {
    "high": torch.tensor([1.0, 0.2, 0.2], dtype=torch.float32),    # Red
    "medium": torch.tensor([1.0, 1.0, 0.2], dtype=torch.float32),  # Yellow
    "low": torch.tensor([0.2, 0.6, 1.0], dtype=torch.float32),     # Blue
}
_STATIC_COLOR = torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32)  # Dark grey


def visualize_capacity_allocation(
    student_configs: List[Dict],
    cluster_mean_scores: List[float],
    tb_writer=None,
    iteration: int = 0,
    logger=None,
) -> None:
    """Visualize capacity allocation across clusters.
    
    This function:
    1. Renders a pseudo-color image showing capacity tiers (high/medium/low)
    2. Logs FLOPs and parameter estimates to TensorBoard
    
    Args:
        student_configs: List of capacity configs for each cluster.
        cluster_mean_scores: Mean dynamic score for each cluster.
        tb_writer: Optional TensorBoard writer.
        iteration: Current training iteration.
        logger: Optional logger instance.
    """
    n_clusters = len(student_configs)
    
    # ── 1. Estimate FLOPs and parameters for each cluster ──
    # Simplified estimation: HexPlane FLOPs ∝ (spatial_res^3 * time_res * feat_dim)
    # MLP FLOPs ∝ (hidden_dim * feat_dim * 3)  (for output d_xyz, d_rot, d_scale)
    cluster_flops = []
    cluster_params = []
    
    for config in student_configs:
        spatial_res = config.get("spatial_resolutions", [64, 128])
        time_res = config.get("time_resolutions", [64, 128])
        feat_dim = config.get("feat_dim", 8)
        mlp_hidden = config.get("mlp_hidden_dim", 48)
        
        # HexPlane FLOPs (simplified: product of resolutions * feat_dim)
        hex_flops = 1.0
        for s_res in spatial_res:
            hex_flops *= s_res
        for t_res in time_res:
            hex_flops *= t_res
        hex_flops *= feat_dim * 3  # 3 levels, fusion
        
        # MLP FLOPs (2-layer: feat_dim -> hidden -> 10 output)
        mlp_flops = feat_dim * mlp_hidden + mlp_hidden * 10
        
        total_flops = hex_flops + mlp_flops
        cluster_flops.append(total_flops)
        
        # Parameter count
        # HexPlane params: sum(resolution * feat_dim) for all planes
        hex_params = sum(spatial_res + time_res) * feat_dim * 3
        # MLP params
        mlp_params = feat_dim * mlp_hidden + mlp_hidden * 10
        total_params = hex_params + mlp_params
        cluster_params.append(total_params)
    
    # Normalize for visualization (0-1 range)
    max_flops = max(cluster_flops) + 1e-8
    max_params = max(cluster_params) + 1e-8
    
    # ── 2. Log to TensorBoard ──
    if tb_writer is not None:
        # Log aggregated stats only (simplified for TensorBoard 2.20.0 compatibility)
        total_flops = sum(cluster_flops)
        total_params = sum(cluster_params)
        avg_flops = total_flops / n_clusters if n_clusters > 0 else 0
        avg_params = total_params / n_clusters if n_clusters > 0 else 0
        
        tb_writer.add_scalar('capacity/total_flops', total_flops, iteration)
        tb_writer.add_scalar('capacity/total_params', total_params, iteration)
        tb_writer.add_scalar('capacity/avg_flops_per_cluster', avg_flops, iteration)
        tb_writer.add_scalar('capacity/avg_params_per_cluster', avg_params, iteration)
    
    # Log summary
    if logger is not None:
        _msg = (f"[CAPACITY] Iteration {iteration}: "
                f"Total FLOPs={total_flops/1e6:.2f}M, "
                f"Total Params={total_params/1e6:.2f}M, "
                f"Avg FLOPs/cluster={avg_flops/1e6:.2f}M")
        logger.info(_msg)
        print(_msg)
        
        # Per-cluster breakdown
        for k in range(n_clusters):
            _msg = (f"[CAPACITY]   Cluster {k}: "
                    f"Score={cluster_mean_scores[k]:.4f}, "
                    f"FLOPs={cluster_flops[k]/1e6:.2f}M, "
                    f"Params={cluster_params[k]/1e6:.2f}M, "
                    f"Spatial={student_configs[k]['spatial_resolutions']}, "
                    f"Time={student_configs[k]['time_resolutions']}")
            logger.info(_msg)
            print(_msg)


def render_capacity_pseudocolor(
    gaussians,
    student_configs: List[Dict],
    cluster_labels: torch.Tensor,
    viewpoint_cam,
    deform,
    pipe,
    bg_color: torch.Tensor,
    mult: float,
    is_6dof: bool = False,
    save_path: Optional[str] = None,
    iteration: int = 0,
) -> torch.Tensor:
    """Render a pseudo-color image where capacity tiers are shown in different colors.
    
    Color scheme:
    - Red: High capacity tier
    - Yellow: Medium capacity tier
    - Blue: Low capacity tier
    - Dark grey: Static Gaussians
    
    The tier label is read directly from student_configs["tier"].
    
    Args:
        gaussians: GaussianModel instance.
        student_configs: Capacity config for each cluster (must include "tier" field).
        cluster_labels: (N,) int32 tensor with cluster assignments (-1 for static).
        viewpoint_cam: Camera for rendering.
        deform: Deformation model.
        pipe: Pipeline parameters.
        bg_color: Background color tensor.
        mult: FastGS mult parameter.
        is_6dof: Whether this is a 6-DoF scene.
        save_path: Path to save the rendered image.
        iteration: Current iteration for logging.
    
    Returns:
        rendered_image: (3, H, W) float tensor.
    """
    import math
    import os
    from diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from utils.rigid_utils import from_homogenous, to_homogenous
    
    N = gaussians.get_xyz.shape[0]
    n_clusters = len(student_configs)
    
    # ── 1. Build tier-to-color mapping from student_configs ──
    # Read tier labels directly from configs (no inference needed)
    # Determine device from cluster_labels
    device = cluster_labels.device
    
    # Map tier labels to colors (on correct device)
    tier_colors_map = {
        "high": _CAPACITY_TIER_COLORS["high"].to(device=device),
        "medium": _CAPACITY_TIER_COLORS["medium"].to(device=device),
        "low": _CAPACITY_TIER_COLORS["low"].to(device=device),
    }
    
    # Assign color to each cluster based on its tier label.
    # When frequency-based allocation provides independent hex_tier / mlp_tier,
    # blend 70% HexPlane tier color + 30% MLP tier color so both dimensions
    # are visible in a single pseudo-color render.
    tier_colors = torch.zeros(n_clusters, 3, dtype=torch.float32, device=device)
    for k in range(n_clusters):
        config = student_configs[k]
        hex_tier = config.get("hex_tier", None)
        mlp_tier = config.get("mlp_tier", None)
        if hex_tier is not None and mlp_tier is not None:
            # Dual-tier blending for frequency-based allocation
            hex_color = tier_colors_map.get(hex_tier, tier_colors_map["low"])
            mlp_color = tier_colors_map.get(mlp_tier, tier_colors_map["low"])
            tier_colors[k] = 0.7 * hex_color + 0.3 * mlp_color
        else:
            tier = config.get("tier", "low")
            tier_colors[k] = tier_colors_map.get(tier, tier_colors_map["low"])
    
    # ── 2. Assign colors to Gaussians ──
    colors = _STATIC_COLOR.to(device=device).unsqueeze(0).expand(N, 3).clone()  # Default: dark grey
    for k in range(n_clusters):
        mask = (cluster_labels == k)
        if mask.any():
            colors[mask] = tier_colors[k]
    
    # ── 3. Compute deformation ──
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
    
    # ── 4. Rasterize ──
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
        sh_degree=0,
        campos=viewpoint_cam.camera_center,
        mult=mult,
        prefiltered=False,
        debug=False,
        get_flag=None,
        metric_map=metric_map,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    with torch.no_grad():
        rendered_image, _, _, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            opacities=opacity,
            colors_precomp=colors,
            scales=scales,
            rotations=rotations,
        )
    
    rendered_image = rendered_image.clamp(0.0, 1.0)
    
    # ── 5. Save ──
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import torchvision
        torchvision.utils.save_image(rendered_image, save_path)
        _msg = f"[CAPACITY-VIS] Capacity allocation render saved to {save_path}"
        if logger is not None:
            logger.info(_msg)
        print(_msg)
    
    return rendered_image


def infer_student_configs_from_weights(
    cluster_tiers: Dict[int, object],
    n_clusters: int,
    capacity_tier_configs: Optional[Dict] = None,
) -> List[Dict]:
    """Infer student configs from weight file tier labels.
    
    This function reconstructs the student configuration list by parsing
    tier labels from weight filenames.  It supports two conventions:
    
    * **Single-tier** (tiered/linear strategy):
        ``cluster_tiers[id] = "high"`` – a plain string looked up in
        ``capacity_tier_configs["tiered"]["tiers"]``.
    * **Dual-tier** (frequency strategy):
        ``cluster_tiers[id] = {"hex_tier": "high", "mlp_tier": "low"}`` – a
        dict whose HexPlane and MLP tiers are looked up independently in
        ``capacity_tier_configs["frequency_based"]``.
    
    Args:
        cluster_tiers: Dict mapping cluster_id to *either* a tier string
            **or** a ``{"hex_tier": ..., "mlp_tier": ...}`` dict.
        n_clusters: Total number of clusters.
        capacity_tier_configs: Full JSON config (must contain ``"tiered"``
            and/or ``"frequency_based"`` keys).
    
    Returns:
        List of student configs, one per cluster.
    """
    if capacity_tier_configs is None:
        raise ValueError("[ERROR] capacity_tier_configs is None!")

    # Detect whether any entry uses dual-tier (frequency) format
    has_freq = any(
        isinstance(v, dict) for v in cluster_tiers.values()
    )

    if has_freq:
        # ── Frequency-based dual-tier reconstruction ──
        if "frequency_based" not in capacity_tier_configs:
            raise ValueError(
                "[ERROR] Dual-tier weight filenames detected but "
                "'frequency_based' section missing from capacity_tier_configs.json"
            )
        fb_cfg = capacity_tier_configs["frequency_based"]
        hex_tiers_cfg = fb_cfg["hex_tiers"]
        mlp_tiers_cfg = fb_cfg["mlp_tiers"]

        student_configs: List[Dict] = []
        for cluster_id in range(n_clusters):
            entry = cluster_tiers.get(cluster_id)
            if entry is not None and isinstance(entry, dict):
                h_tier = entry.get("hex_tier", "medium")
                m_tier = entry.get("mlp_tier", "medium")
                h_cfg = hex_tiers_cfg.get(h_tier, hex_tiers_cfg["medium"])
                m_cfg = mlp_tiers_cfg.get(m_tier, mlp_tiers_cfg["medium"])

                config = {
                    "spatial_resolutions": list(h_cfg["spatial_resolutions"]),
                    "time_resolutions": list(h_cfg["time_resolutions"]),
                    "feat_dim": h_cfg["feat_dim"],
                    "mlp_hidden_dim": m_cfg["mlp_hidden_dim"],
                    "mlp_layer_num": m_cfg.get("mlp_layer_num", 2),
                    "hex_tier": h_tier,
                    "mlp_tier": m_tier,
                    "tier": h_tier,  # backward-compat label
                }
                student_configs.append(config)
            else:
                # Fallback: treat as single tier string
                tier = entry if isinstance(entry, str) else "medium"
                h_cfg = hex_tiers_cfg.get(tier, hex_tiers_cfg["medium"])
                m_cfg = mlp_tiers_cfg.get(tier, mlp_tiers_cfg["medium"])
                config = {
                    "spatial_resolutions": list(h_cfg["spatial_resolutions"]),
                    "time_resolutions": list(h_cfg["time_resolutions"]),
                    "feat_dim": h_cfg["feat_dim"],
                    "mlp_hidden_dim": m_cfg["mlp_hidden_dim"],
                    "mlp_layer_num": m_cfg.get("mlp_layer_num", 2),
                    "hex_tier": tier,
                    "mlp_tier": tier,
                    "tier": tier,
                }
                student_configs.append(config)
                print(f"[WARNING] Cluster {cluster_id}: missing dual-tier info, "
                      f"falling back to '{tier}' for both")
        return student_configs

    # ── Single-tier reconstruction (original tiered / linear) ──
    if "tiered" not in capacity_tier_configs:
        raise ValueError("[ERROR] 'tiered' section not found in capacity_tier_configs!")
    default_tiers = capacity_tier_configs["tiered"]["tiers"]

    student_configs = []
    for cluster_id in range(n_clusters):
        if cluster_id in cluster_tiers:
            tier = cluster_tiers[cluster_id]
            if tier in default_tiers:
                config = default_tiers[tier].copy()
                config["tier"] = tier
                student_configs.append(config)
            else:
                print(f"[WARNING] Unknown tier '{tier}' for cluster {cluster_id}, using medium config")
                config = default_tiers["medium"].copy()
                config["tier"] = "medium"
                student_configs.append(config)
        else:
            print(f"[WARNING] No weight file found for cluster {cluster_id}, using default config")
            config = default_tiers["medium"].copy()
            config["tier"] = "medium"
            student_configs.append(config)
    
    return student_configs


# ---------------------------------------------------------------------------
# Frequency-based capacity analysis (plan_frequency_capacity.md)
# ---------------------------------------------------------------------------

def analyze_cluster_capacity_needs(
    gaussians,
    deform,
    cluster_labels: torch.Tensor,
    n_clusters: int,
    n_time_samples: int = 32,
) -> Dict[str, List]:
    """Compute per-cluster temporal complexity and motion heterogeneity in one pass.

    Shares teacher inference across both metrics to avoid redundant forward passes.

    **Metric A – temporal_complexity** (RMS acceleration):
        Acceleration = d²(displacement)/dt².  In the frequency domain this is
        proportional to ω² |F(ω)|, so a large RMS acceleration indicates that
        the motion signal has significant high-frequency energy and therefore
        requires higher HexPlane resolution (Nyquist bandwidth).

    **Metric B – heterogeneity** (intra-cluster trajectory variance):
        Each Gaussian's displacement trajectory is flattened into a (T×3)-dim
        feature vector.  The mean squared distance from each trajectory to the
        cluster centroid measures how diverse the motion patterns are within
        the cluster.  High heterogeneity → MLP needs more capacity to
        discriminate different motion modes.

    Args:
        gaussians: GaussianModel instance (must have ``get_xyz``).
        deform: Teacher deformation network (``DeformModel_4DGS`` or
            ``ClusteredDeformModel`` with ``step_teacher``).
        cluster_labels: ``(N,)`` int32 tensor, -1 = static, 0..K-1 = cluster.
        n_clusters: Number of clusters K.
        n_time_samples: Number of uniform time steps for teacher sampling.

    Returns:
        Dict with keys:
          - ``"temporal_complexity"``: ``List[float]`` of length *n_clusters*.
          - ``"heterogeneity"``: ``List[float]`` of length *n_clusters*.
          - ``"n_gaussians"``: ``List[int]``, number of Gaussians per cluster.
    """
    xyz = gaussians.get_xyz.detach()  # (N, 3)
    N = xyz.shape[0]
    time_steps = torch.linspace(0, 1, n_time_samples, device=xyz.device)

    # ── Shared teacher inference ──
    displacements: List[torch.Tensor] = []
    with torch.no_grad():
        for t_val in time_steps:
            t_input = t_val.unsqueeze(0).expand(N, 1)  # (N, 1)
            if hasattr(deform, 'step_teacher'):
                d_xyz, _, _ = deform.step_teacher(xyz, t_input)
            elif hasattr(deform, 'step'):
                d_xyz, _, _ = deform.step(xyz, t_input)
            else:
                d_xyz = deform.deform(xyz, t_input)[0]

            if not torch.is_tensor(d_xyz):
                d_xyz = torch.zeros_like(xyz)
            displacements.append(d_xyz)

    disp = torch.stack(displacements, dim=0)  # (T, N, 3)

    # ── Metric A: temporal complexity (RMS acceleration) ──
    velocity = disp[1:] - disp[:-1]              # (T-1, N, 3)
    acceleration = velocity[1:] - velocity[:-1]   # (T-2, N, 3)
    # per-Gaussian RMS acceleration magnitude
    per_gaussian_accel = acceleration.pow(2).mean(dim=0).sum(dim=-1).sqrt()  # (N,)

    # ── Metric B: trajectory feature for heterogeneity ──
    traj_feat = disp.permute(1, 0, 2).reshape(N, -1)  # (N, T*3)

    # ── Aggregate per cluster ──
    temporal_complexity: List[float] = []
    heterogeneity: List[float] = []
    n_gaussians_list: List[int] = []

    for k in range(n_clusters):
        mask = (cluster_labels == k)
        count = int(mask.sum().item())
        n_gaussians_list.append(count)

        if count == 0:
            temporal_complexity.append(0.0)
            heterogeneity.append(0.0)
            continue

        # Metric A
        temporal_complexity.append(float(per_gaussian_accel[mask].mean().item()))

        # Metric B
        if count < 2:
            heterogeneity.append(0.0)
        else:
            cluster_traj = traj_feat[mask]  # (M_k, T*3)
            centroid = cluster_traj.mean(dim=0, keepdim=True)
            intra_var = float((cluster_traj - centroid).pow(2).mean().item())
            heterogeneity.append(intra_var)

    logger.info(
        "[FREQ-CAPACITY] Temporal complexity: %s",
        [f"{v:.6f}" for v in temporal_complexity],
    )
    logger.info(
        "[FREQ-CAPACITY] Heterogeneity: %s",
        [f"{v:.6f}" for v in heterogeneity],
    )

    return {
        "temporal_complexity": temporal_complexity,
        "heterogeneity": heterogeneity,
        "n_gaussians": n_gaussians_list,
    }


def allocate_capacity_by_frequency(
    temporal_complexity: List[float],
    heterogeneity: List[float],
    n_clusters: int,
    capacity_tier_configs: Dict,
    strategy: str = "independent_tiered",
) -> List[Dict]:
    """Frequency-driven capacity allocation (independent HexPlane / MLP tiers).

    Unlike :func:`allocate_capacity_by_score` which uses a single dynamic-score
    ranking for all parameters, this function ranks clusters independently:

    * **HexPlane resolution tier** ← ``temporal_complexity`` ranking
      (high-frequency motion needs higher Nyquist bandwidth)
    * **MLP hidden width tier** ← ``heterogeneity`` ranking
      (diverse intra-cluster motion needs more expressive decoder)

    This decoupling allows combinations such as *High HexPlane + Low MLP*
    (fast but uniform motion) or *Low HexPlane + High MLP* (slow but diverse
    motion) that were impossible with the old single-score scheme.

    Args:
        temporal_complexity: Per-cluster Metric A values.
        heterogeneity: Per-cluster Metric B values.
        n_clusters: Number of clusters K.
        capacity_tier_configs: Full JSON config dict (must contain
            ``"frequency_based"`` key).
        strategy: ``"independent_tiered"`` (recommended) — HexPlane and MLP
            tiers assigned independently.

    Returns:
        ``student_configs``: list of K dicts, each compatible with
        ``ClusteredDeformModel.__init__``.
    """
    if strategy != "independent_tiered":
        raise ValueError(
            f"Unknown frequency strategy: {strategy!r}. "
            "Currently only 'independent_tiered' is supported."
        )

    fb_cfg = capacity_tier_configs["frequency_based"]
    hex_tiers_cfg = fb_cfg["hex_tiers"]
    mlp_tiers_cfg = fb_cfg["mlp_tiers"]

    n_high = max(1, n_clusters // 3)

    # ── HexPlane tier by temporal_complexity (descending) ──
    hex_sorted = sorted(
        range(n_clusters),
        key=lambda i: temporal_complexity[i],
        reverse=True,
    )
    hex_tier: Dict[int, str] = {}
    for rank, cid in enumerate(hex_sorted):
        if rank < n_high:
            hex_tier[cid] = "high"
        elif rank < 2 * n_high:
            hex_tier[cid] = "medium"
        else:
            hex_tier[cid] = "low"

    # ── MLP tier by heterogeneity (descending) ──
    mlp_sorted = sorted(
        range(n_clusters),
        key=lambda i: heterogeneity[i],
        reverse=True,
    )
    mlp_tier: Dict[int, str] = {}
    for rank, cid in enumerate(mlp_sorted):
        if rank < n_high:
            mlp_tier[cid] = "high"
        elif rank < 2 * n_high:
            mlp_tier[cid] = "medium"
        else:
            mlp_tier[cid] = "low"

    # ── Compose student configs ──
    student_configs: List[Dict] = []
    for k in range(n_clusters):
        h_tier = hex_tier[k]
        m_tier = mlp_tier[k]
        h_cfg = hex_tiers_cfg[h_tier]
        m_cfg = mlp_tiers_cfg[m_tier]

        config = {
            "spatial_resolutions": list(h_cfg["spatial_resolutions"]),
            "time_resolutions": list(h_cfg["time_resolutions"]),
            "feat_dim": h_cfg["feat_dim"],
            "mlp_hidden_dim": m_cfg["mlp_hidden_dim"],
            "mlp_layer_num": m_cfg.get("mlp_layer_num", 2),
            "hex_tier": h_tier,
            "mlp_tier": m_tier,
            "tier": h_tier,  # backward-compat: use HexPlane tier as overall label
        }
        student_configs.append(config)

        logger.info(
            "[FREQ-CAPACITY] Cluster %d: hex_tier=%s (complexity=%.6f), "
            "mlp_tier=%s (heterogeneity=%.6f) → spatial=%s, feat=%d, mlp_hidden=%d",
            k, h_tier, temporal_complexity[k],
            m_tier, heterogeneity[k],
            config["spatial_resolutions"], config["feat_dim"],
            config["mlp_hidden_dim"],
        )

    return student_configs
