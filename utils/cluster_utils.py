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
from typing import Dict, Optional, Tuple

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
    save_path: Optional[str] = None,
) -> Dict[str, object]:
    """Cluster dynamic Gaussians and return results.

    Args:
        gaussians: GaussianModel instance.
        dynamic_thresh: Probability threshold for selecting dynamic Gaussians.
        n_clusters: Number of KMeans clusters.
        w_xyz / w_color / w_motion: Feature weights.
        temperature: Sigmoid temperature for dynamic prob.
        tb_writer: Optional TensorBoard writer.
        iteration: Current training iteration (for logging).
        save_path: If provided, save cluster labels as .npz.

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
    with torch.no_grad():
        if hasattr(gaussians, 'get_dynamic_prob_t'):
            dyn_prob = gaussians.get_dynamic_prob_t(temperature).squeeze(-1)  # (N,)
        else:
            dyn_prob = gaussians.get_dynamic_prob.squeeze(-1)
        dynamic_mask = dyn_prob > dynamic_thresh  # boolean (N,)
        n_dynamic = int(dynamic_mask.sum().item())

    _msg = f"[CLUSTER] Selecting dynamic Gaussians: {n_dynamic}/{N} (thresh={dynamic_thresh:.2f})"
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

    # ── 7. Save results ──
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path,
            labels=labels_full.cpu().numpy(),
            dynamic_mask=dynamic_mask.cpu().numpy(),
            metrics_silhouette=sil,
            metrics_intra=intra,
            metrics_inter=inter,
            metrics_ratio=ratio,
            cluster_counts=cluster_counts,
        )
        _msg = f"[CLUSTER] Results saved to {save_path}"
        logger.info(_msg)
        print(_msg)

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
