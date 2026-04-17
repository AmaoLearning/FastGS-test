#!/usr/bin/env python3
"""Estimate teacher HexPlane spatio-temporal resolutions using Structure Function.

Direction E implementation: Flow + DA3 Depth + Structure Function.
Extends v1 (test_flow_resolution.py) with depth-based 3D back-projection and
structure function analysis for robust spatial resolution estimation.

Outputs:
  1. stdout summary table
  2. ``<output_path>/flow_resolution_analysis_v2.log``
  3. ``<output_path>/resolution_config_v2.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

# ---------------------------------------------------------------------------
# Reuse v1 functions
# ---------------------------------------------------------------------------
from scripts.test_flow_resolution import (  # noqa: E402
    LazyCameraInfo,
    CameraSequence,
    SceneStats,
    read_n3v_cameras_lazy,
    fetch_ply,
    compute_scene_stats,
    load_training_cameras,
    group_sequences,
    forward_backward_consistency_check,
    load_flow_with_mask,
    subsample_sequence,
    stack_sequence,
    masked_temporal_fill,
    percentile_cutoff,
    analyze_temporal_cutoff,
    analyze_gradient_ratio,
    accumulate_spatial_spectrum,
    build_radial_bins,
    snap_to_candidate,
    estimate_plane_params,
    configure_logging as _configure_logging_v1,
    parse_candidates,
)

logger = logging.getLogger("flow_resolution_v2")

# ═══════════════════════════════════════════════════════════════════════════
# New data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DepthAlignResult:
    """Result of RANSAC affine depth alignment."""
    alpha: float = 0.0
    beta: float = 0.0
    inlier_rmse: float = float("inf")
    inlier_ratio: float = 0.0
    n_anchors: int = 0
    valid: bool = False


@dataclass
class StructureFunctionResult:
    """Result of structure function analysis."""
    D_iso: Optional[torch.Tensor] = None      # [n_bins]
    D_x: Optional[torch.Tensor] = None        # [n_bins]
    D_y: Optional[torch.Tensor] = None        # [n_bins]
    D_z: Optional[torch.Tensor] = None        # [n_bins]
    r_bin_centers: Optional[torch.Tensor] = None  # [n_bins]
    counts_iso: Optional[torch.Tensor] = None     # [n_bins]
    D_inf: float = 0.0
    D_0: float = 0.0
    r_c: float = 0.0
    f_c: float = 0.0
    Rs_raw: float = 0.0
    Rs_snapped: int = 0
    n_points_total: int = 0
    n_sub_actual: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Depth discovery & loading
# ═══════════════════════════════════════════════════════════════════════════


def discover_depth_paths(
    source_path: str,
    cam_infos: List[LazyCameraInfo],
    depth_dir_name: str = "depth",
) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """Discover DA3 depth + confidence paths for each camera frame.

    Args:
        source_path: N3V scene root directory.
        cam_infos: Camera list from :func:`read_n3v_cameras_lazy`.
        depth_dir_name: Subdirectory name relative to *source_path*.

    Returns:
        Mapping ``uid -> (depth_path | None, conf_path | None)``.
    """
    depth_root = os.path.join(source_path, depth_dir_name)
    result: Dict[int, Tuple[Optional[str], Optional[str]]] = {}

    for cam in cam_infos:
        cam_name = Path(cam.image_path).parent.parent.name  # e.g. "cam01"
        stem = Path(cam.image_path).stem                     # e.g. "0000"
        dp = os.path.join(depth_root, cam_name, f"depth_{stem}.npy")
        cp = os.path.join(depth_root, cam_name, f"conf_{stem}.npy")
        result[cam.uid] = (
            dp if os.path.isfile(dp) else None,
            cp if os.path.isfile(cp) else None,
        )
    return result


def load_depth_map(path: str, device: torch.device) -> torch.Tensor:
    """Load a ``.npy`` depth map as a float32 tensor.

    Returns:
        ``[H, W]`` float32 tensor (DA3 relative or metric depth).
    """
    arr = np.load(path)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Projection utilities
# ═══════════════════════════════════════════════════════════════════════════


def _project_points_to_camera(
    points_3d: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    focal: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space 3D points to camera pixel coordinates.

    Returns:
        uv: ``[K, 2]`` pixel coordinates of visible points.
        z_metric: ``[K]`` camera-frame Z values.
        valid_mask: ``[M]`` bool mask.
    """
    # p_cam = R^T @ p_world + T   (getWorld2View convention)
    p_cam = points_3d @ R + T.unsqueeze(0)  # [M, 3]  (equivalent: (R^T @ p.T).T + T)

    z = p_cam[:, 2]
    u = focal * p_cam[:, 0] / z + cx
    v = focal * p_cam[:, 1] / z + cy

    valid = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    uv = torch.stack([u[valid], v[valid]], dim=1)  # [K, 2]
    return uv, z[valid], valid


def sample_depth_at_pixels(
    depth_map: torch.Tensor,
    uv: torch.Tensor,
) -> torch.Tensor:
    """Bilinear-interpolate depth map at sub-pixel locations.

    Args:
        depth_map: ``[H, W]`` depth values.
        uv: ``[K, 2]`` pixel coordinates (u=col, v=row).

    Returns:
        ``[K]`` interpolated depth values.
    """
    H, W = depth_map.shape
    # Normalize to [-1, 1] for grid_sample (expects [N, C, H, W])
    grid_x = 2.0 * uv[:, 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * uv[:, 1] / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]

    depth_4d = depth_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    sampled = F.grid_sample(
        depth_4d, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return sampled.squeeze()  # [K]


# ═══════════════════════════════════════════════════════════════════════════
# Affine depth alignment (RANSAC)
# ═══════════════════════════════════════════════════════════════════════════


def affine_align_depth(
    depth_rel: torch.Tensor,
    points_3d: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    focal: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    ransac_iters: int = 100,
    inlier_thresh: float = 0.1,
    min_anchors: int = 10,
) -> DepthAlignResult:
    """RANSAC affine alignment: ``Z_metric = alpha * d_rel + beta``.

    Uses SfM 3D points as anchor correspondences between known metric depth
    and DA3 relative depth.

    Args:
        depth_rel: ``[H, W]`` DA3 relative depth map.
        points_3d: ``[M, 3]`` SfM world-space points.
        R, T: Camera extrinsics.
        focal, cx, cy, width, height: Camera intrinsics.
        ransac_iters: Number of RANSAC iterations.
        inlier_thresh: Absolute residual threshold for inliers.
        min_anchors: Minimum visible anchors required.

    Returns:
        :class:`DepthAlignResult` with fitted ``alpha, beta`` and diagnostics.
    """
    uv, z_metric, valid_mask = _project_points_to_camera(
        points_3d, R, T, focal, cx, cy, width, height,
    )
    if uv.shape[0] < min_anchors:
        return DepthAlignResult(n_anchors=uv.shape[0], valid=False)

    d_rel = sample_depth_at_pixels(depth_rel, uv)  # [K]

    # Filter valid (positive on both sides)
    ok = (z_metric > 0.01) & (d_rel.abs() > 1e-6)
    z_metric = z_metric[ok]
    d_rel = d_rel[ok]
    if z_metric.shape[0] < min_anchors:
        return DepthAlignResult(n_anchors=int(z_metric.shape[0]), valid=False)

    N = z_metric.shape[0]
    best_n_inliers = 0
    best_alpha = 1.0
    best_beta = 0.0

    for _ in range(ransac_iters):
        idx = torch.randint(0, N, (2,), device=z_metric.device)
        if idx[0] == idx[1]:
            continue
        d0, d1 = d_rel[idx[0]], d_rel[idx[1]]
        z0, z1 = z_metric[idx[0]], z_metric[idx[1]]
        denom = d0 - d1
        if abs(float(denom.item())) < 1e-10:
            continue
        alpha = (z0 - z1) / denom
        beta = z0 - alpha * d0

        residuals = (alpha * d_rel + beta - z_metric).abs()
        inlier_mask = residuals < inlier_thresh
        n_in = int(inlier_mask.sum().item())
        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best_alpha = float(alpha.item())
            best_beta = float(beta.item())

    # Refit on best inlier set
    residuals = (best_alpha * d_rel + best_beta - z_metric).abs()
    inlier_mask = residuals < inlier_thresh
    if inlier_mask.sum() >= 2:
        d_in = d_rel[inlier_mask]
        z_in = z_metric[inlier_mask]
        # Least-squares: z = alpha*d + beta
        A = torch.stack([d_in, torch.ones_like(d_in)], dim=1)  # [K', 2]
        # Normal equations
        sol = torch.linalg.lstsq(A, z_in.unsqueeze(1)).solution.squeeze()
        best_alpha = float(sol[0].item())
        best_beta = float(sol[1].item())
        final_residuals = (best_alpha * d_in + best_beta - z_in)
        inlier_rmse = float(final_residuals.pow(2).mean().sqrt().item())
    else:
        inlier_rmse = float("inf")

    # Sign check: alpha should be positive (deeper = larger depth value)
    if best_alpha < 0:
        logger.warning("Negative alpha=%.4f detected; flipping depth polarity.", best_alpha)
        best_alpha = -best_alpha
        best_beta = -best_beta
        # Re-run on flipped
        residuals = (best_alpha * (-d_rel) + best_beta - z_metric).abs()
        inlier_mask = residuals < inlier_thresh
        best_n_inliers = int(inlier_mask.sum().item())

    return DepthAlignResult(
        alpha=best_alpha,
        beta=best_beta,
        inlier_rmse=inlier_rmse,
        inlier_ratio=best_n_inliers / max(N, 1),
        n_anchors=N,
        valid=best_n_inliers >= min_anchors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3D back-projection
# ═══════════════════════════════════════════════════════════════════════════


def unproject_pixels(
    uv: torch.Tensor,
    depth: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    focal: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Back-project pixels + depth to world coordinates.

    Inverse of :func:`_project_points_to_camera`:
        ``p_cam = Z * K_inv @ [u, v, 1]^T``
        ``p_world = R @ (p_cam - T)``

    Args:
        uv: ``[N, 2]`` pixel coordinates (u=col, v=row).
        depth: ``[N]`` metric depth (camera-frame Z).
        R: ``[3, 3]`` rotation matrix.
        T: ``[3]`` translation vector.
        focal: Pixel focal length.
        cx, cy: Principal point.

    Returns:
        ``[N, 3]`` world coordinates.
    """
    x_cam = (uv[:, 0] - cx) / focal * depth
    y_cam = (uv[:, 1] - cy) / focal * depth
    z_cam = depth
    p_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)  # [N, 3]

    # p_world = R @ (p_cam - T)
    p_world = (p_cam - T.unsqueeze(0)) @ R.T  # [N, 3]: (p_cam - T) @ R^T == R @ (p_cam - T)^T transposed
    return p_world


# ═══════════════════════════════════════════════════════════════════════════
# Scene flow per frame pair
# ═══════════════════════════════════════════════════════════════════════════


def compute_scene_flow_frame(
    cam_info: LazyCameraInfo,
    depth_path_t: Optional[str],
    depth_path_tp1: Optional[str],
    points_3d: torch.Tensor,
    scene_stats: SceneStats,
    device: torch.device,
    focal: float,
    cx: float,
    cy: float,
    flow_magnitude_thresh: float,
    use_consistency_mask: bool,
    subsample_factor: int,
    ransac_iters: int,
    inlier_thresh: float,
    min_anchors: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, DepthAlignResult]]:
    """Compute 3D scene flow for a single frame pair (t, t+1) on one camera.

    N3V cameras are static, so R, T are the same for frame t and t+1.

    Returns:
        (positions_world, displacements_world, align_result) or None if invalid.
        positions_world: ``[N_valid, 3]`` world coordinates of source pixels.
        displacements_world: ``[N_valid, 3]`` 3D displacement vectors.
    """
    if depth_path_t is None or depth_path_tp1 is None:
        return None
    if cam_info.flow_fwd_path is None:
        return None

    R = torch.from_numpy(cam_info.R.astype(np.float32)).to(device)
    T = torch.from_numpy(cam_info.T.astype(np.float32)).to(device)
    W, H = cam_info.width, cam_info.height

    # 1. Load flow + mask
    flow_fwd, mask = load_flow_with_mask(
        cam_info, device, flow_magnitude_thresh, use_consistency_mask,
    )
    # flow_fwd: [2, H, W], mask: [1, H, W]

    # 2. Load depth maps
    depth_t = load_depth_map(depth_path_t, device)     # [H, W]
    depth_tp1 = load_depth_map(depth_path_tp1, device)  # [H, W]

    # 3. Affine-align depth_t with SfM
    align_t = affine_align_depth(
        depth_t, points_3d, R, T, focal, cx, cy, W, H,
        ransac_iters=ransac_iters, inlier_thresh=inlier_thresh,
        min_anchors=min_anchors,
    )
    if not align_t.valid:
        return None

    # Apply alignment
    depth_t_metric = align_t.alpha * depth_t + align_t.beta  # [H, W]
    # For tp1 on same camera, reuse same alignment (static camera, small time step)
    depth_tp1_metric = align_t.alpha * depth_tp1 + align_t.beta

    # 4. Build pixel grid (subsampled)
    ys = torch.arange(0, H, subsample_factor, device=device, dtype=torch.float32)
    xs = torch.arange(0, W, subsample_factor, device=device, dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(ys, xs, indexing="ij")
    grid_uv = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)  # [N_pix, 2]

    # Sample quantities at subsampled grid
    flow_sub = flow_fwd[:, ::subsample_factor, ::subsample_factor]   # [2, Hs, Ws]
    mask_sub = mask[:, ::subsample_factor, ::subsample_factor]        # [1, Hs, Ws]
    depth_t_sub = depth_t_metric[::subsample_factor, ::subsample_factor]   # [Hs, Ws]

    flow_flat = flow_sub.reshape(2, -1).T                 # [N_pix, 2]
    mask_flat = mask_sub.reshape(-1) > 0.5                # [N_pix]
    depth_t_flat = depth_t_sub.reshape(-1)                # [N_pix]

    # Target pixel locations
    uv_target = grid_uv + flow_flat  # [N_pix, 2]

    # Sample depth at target locations from depth_tp1_metric
    depth_tp1_at_target = sample_depth_at_pixels(depth_tp1_metric, uv_target)  # [N_pix]

    # Combined validity mask
    valid = (
        mask_flat
        & (depth_t_flat > 0.01)
        & (depth_tp1_at_target > 0.01)
        & (uv_target[:, 0] >= 0) & (uv_target[:, 0] < W)
        & (uv_target[:, 1] >= 0) & (uv_target[:, 1] < H)
    )

    if valid.sum() < 10:
        return None

    # 5. Back-project source pixels
    p_world_t = unproject_pixels(
        grid_uv[valid], depth_t_flat[valid], R, T, focal, cx, cy,
    )  # [N_valid, 3]

    # 6. Back-project target pixels
    p_world_tp1 = unproject_pixels(
        uv_target[valid], depth_tp1_at_target[valid], R, T, focal, cx, cy,
    )  # [N_valid, 3]

    # 7. Scene flow = displacement in world space
    displacement = p_world_tp1 - p_world_t  # [N_valid, 3]

    return p_world_t, displacement, align_t


# ═══════════════════════════════════════════════════════════════════════════
# Canonical space normalization
# ═══════════════════════════════════════════════════════════════════════════


def to_canonical(
    positions: torch.Tensor,
    displacements: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform world coordinates to canonical ``[-1, 1]^3``.

    Args:
        positions: ``[N, 3]`` world coordinates.
        displacements: ``[N, 3]`` world-space displacement vectors.
        aabb_min, aabb_max: ``[3]`` scene AABB bounds.

    Returns:
        positions_canon: ``[N', 3]`` in ``[-1, 1]^3`` (outliers filtered).
        displacements_canon: ``[N', 3]`` scaled displacements.
    """
    extent = (aabb_max - aabb_min).clamp_min(1e-6)
    pos_canon = 2.0 * (positions - aabb_min) / extent - 1.0
    disp_canon = 2.0 * displacements / extent

    # Filter outliers beyond [-1.5, 1.5]
    within = (pos_canon.abs() <= 1.5).all(dim=1)
    return pos_canon[within], disp_canon[within]


# ═══════════════════════════════════════════════════════════════════════════
# Structure function computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_structure_function(
    positions: torch.Tensor,
    displacements: torch.Tensor,
    n_bins: int = 64,
    r_min: float = 0.001,
    r_max: float = 1.0,
    device: torch.device = torch.device("cpu"),
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute isotropic + axial second-order structure functions.

    D(r) = <|Δv(x) - Δv(x')|² | ||x - x'|| ∈ bin(r)>

    Uses chunked computation to avoid OOM on large point sets.

    Args:
        positions: ``[N, 3]`` canonical coordinates.
        displacements: ``[N, 3]`` canonical displacement vectors.
        n_bins: Number of radial bins.
        r_min, r_max: Min/max separation distance.
        device: Computation device.
        chunk_size: Chunk size for pairwise computation.

    Returns:
        D_iso: ``[n_bins]`` isotropic structure function.
        D_x, D_y, D_z: ``[n_bins]`` axial structure functions.
        r_centers: ``[n_bins]`` bin centre distances.
        counts: ``[n_bins]`` pair counts per bin.
    """
    N = positions.shape[0]
    positions = positions.to(device)
    displacements = displacements.to(device)

    # Log-spaced bins for better resolution at small scales
    log_edges = torch.linspace(
        math.log(max(r_min, 1e-6)), math.log(r_max), n_bins + 1, device=device,
    )
    edges = log_edges.exp()
    centers = 0.5 * (edges[:-1] + edges[1:])

    D_iso_sum = torch.zeros(n_bins, device=device, dtype=torch.float64)
    D_x_sum = torch.zeros(n_bins, device=device, dtype=torch.float64)
    D_y_sum = torch.zeros(n_bins, device=device, dtype=torch.float64)
    D_z_sum = torch.zeros(n_bins, device=device, dtype=torch.float64)
    counts_iso = torch.zeros(n_bins, device=device, dtype=torch.int64)
    counts_x = torch.zeros(n_bins, device=device, dtype=torch.int64)
    counts_y = torch.zeros(n_bins, device=device, dtype=torch.int64)
    counts_z = torch.zeros(n_bins, device=device, dtype=torch.int64)

    for i in range(0, N, chunk_size):
        pos_i = positions[i:i + chunk_size]    # [Ci, 3]
        disp_i = displacements[i:i + chunk_size]

        for j in range(i, N, chunk_size):
            pos_j = positions[j:j + chunk_size]    # [Cj, 3]
            disp_j = displacements[j:j + chunk_size]

            # Pairwise distances: [Ci, Cj]
            diff_pos = pos_i.unsqueeze(1) - pos_j.unsqueeze(0)     # [Ci, Cj, 3]
            dist = diff_pos.norm(dim=2)                             # [Ci, Cj]

            # Pairwise displacement difference squared
            diff_disp = disp_i.unsqueeze(1) - disp_j.unsqueeze(0)  # [Ci, Cj, 3]
            sq_diff = (diff_disp ** 2).sum(dim=2)                   # [Ci, Cj]

            # Avoid self-pairs when i == j
            if i == j:
                diag_mask = torch.eye(pos_i.shape[0], pos_j.shape[0],
                                      device=device, dtype=torch.bool)
                dist = dist.masked_fill(diag_mask, -1.0)

            # Bin assignment
            flat_dist = dist.reshape(-1)
            flat_sq = sq_diff.reshape(-1).to(torch.float64)
            flat_diff_pos = diff_pos.reshape(-1, 3)

            in_range = (flat_dist >= edges[0]) & (flat_dist < edges[-1])
            if not in_range.any():
                continue

            flat_dist_in = flat_dist[in_range]
            flat_sq_in = flat_sq[in_range]
            flat_dp_in = flat_diff_pos[in_range].abs()

            bin_idx = torch.bucketize(flat_dist_in.log(), log_edges, right=False) - 1
            bin_idx = bin_idx.clamp(0, n_bins - 1)

            D_iso_sum.scatter_add_(0, bin_idx, flat_sq_in)
            counts_iso.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.int64))

            # Axial: bin by |Δx_axis|, measure full ||Δv||²
            for axis, (D_sum, c_sum) in enumerate(
                [(D_x_sum, counts_x), (D_y_sum, counts_y), (D_z_sum, counts_z)]
            ):
                ax_dist = flat_dp_in[:, axis]
                ax_in = (ax_dist >= edges[0]) & (ax_dist < edges[-1])
                if not ax_in.any():
                    continue
                ax_bin = torch.bucketize(ax_dist[ax_in].log(), log_edges, right=False) - 1
                ax_bin = ax_bin.clamp(0, n_bins - 1)
                # Use full displacement squared norm for axial SF
                D_sum.scatter_add_(0, ax_bin, flat_sq_in[ax_in])
                c_sum.scatter_add_(0, ax_bin, torch.ones_like(ax_bin, dtype=torch.int64))

    # Average
    valid_counts = counts_iso.clamp_min(1).to(torch.float64)
    D_iso = (D_iso_sum / valid_counts).to(torch.float32)
    D_x = (D_x_sum / counts_x.clamp_min(1).to(torch.float64)).to(torch.float32)
    D_y = (D_y_sum / counts_y.clamp_min(1).to(torch.float64)).to(torch.float32)
    D_z = (D_z_sum / counts_z.clamp_min(1).to(torch.float64)).to(torch.float32)

    return D_iso, D_x, D_y, D_z, centers, counts_iso


# ═══════════════════════════════════════════════════════════════════════════
# Plateau detection & resolution conversion
# ═══════════════════════════════════════════════════════════════════════════


def estimate_plateau_and_cutoff(
    D: torch.Tensor,
    r_centers: torch.Tensor,
    counts: torch.Tensor,
    plateau_pct: float = 0.95,
    min_count_per_bin: int = 50,
) -> Tuple[float, float, float]:
    """Estimate plateau value, noise floor, and cutoff scale.

    Args:
        D: ``[n_bins]`` structure function values.
        r_centers: ``[n_bins]`` bin centre distances.
        counts: ``[n_bins]`` pair counts.
        plateau_pct: Fraction of ``(D_inf - D_0)`` at which to declare cutoff.
        min_count_per_bin: Minimum pairs for a bin to be considered reliable.

    Returns:
        D_inf: Plateau value.
        D_0: Noise floor.
        r_c: Cutoff scale.
    """
    # Filter unreliable bins
    reliable = counts >= min_count_per_bin
    if reliable.sum() < 3:
        logger.warning("Too few reliable SF bins (%d); using all bins.", int(reliable.sum()))
        reliable = torch.ones_like(counts, dtype=torch.bool)

    D_rel = D[reliable].cpu().numpy()
    r_rel = r_centers[reliable].cpu().numpy()

    if len(D_rel) < 3:
        return 0.0, 0.0, float(r_centers[-1].item())

    # D_inf: mean of top 10% distance bins
    n_top = max(1, len(D_rel) // 10)
    D_inf = float(np.mean(D_rel[-n_top:]))

    # D_0: smallest-distance bin value
    D_0 = float(D_rel[0])

    if D_inf <= D_0 or D_inf <= 0:
        logger.warning("Non-monotonic SF (D_inf=%.4e, D_0=%.4e); returning max r.", D_inf, D_0)
        return D_inf, D_0, float(r_rel[-1])

    if D_0 / D_inf > 0.5:
        logger.warning("High noise floor: D_0/D_inf = %.3f", D_0 / D_inf)

    # Normalized SF
    D_norm = (D_rel - D_0) / (D_inf - D_0)

    # Find first r where D_norm >= plateau_pct
    above = np.where(D_norm >= plateau_pct)[0]
    if len(above) > 0:
        r_c = float(r_rel[above[0]])
    else:
        logger.warning("SF never reaches %.0f%% plateau; using max r.", plateau_pct * 100)
        r_c = float(r_rel[-1])

    return D_inf, D_0, r_c


def cutoff_to_resolution(
    r_c: float,
    nyquist_margin: float,
    candidates: List[int],
) -> Tuple[float, float, int]:
    """Convert cutoff scale to spatial resolution.

    f_c = 1 / (2 * r_c)   (Nyquist frequency corresponding to cutoff scale)
    Rs_raw = nyquist_margin * 2 * f_c * L + 1
           = nyquist_margin / r_c * L + 1

    For canonical space L=2 (range [-1,1]):
        Rs_raw = 2 * nyquist_margin / r_c + 1

    Returns:
        f_c: Cutoff frequency.
        Rs_raw: Continuous resolution estimate.
        Rs_snapped: Snapped to candidate set.
    """
    if r_c <= 1e-8:
        logger.warning("r_c=%.2e is too small; defaulting to max candidate.", r_c)
        return 0.0, float(candidates[-1]), candidates[-1]

    f_c = 1.0 / (2.0 * r_c)
    # L=2 for canonical [-1,1]
    Rs_raw = 2.0 * nyquist_margin / r_c + 1.0
    Rs_snapped = snap_to_candidate(Rs_raw, candidates)

    return f_c, Rs_raw, Rs_snapped


# ═══════════════════════════════════════════════════════════════════════════
# Scatter point collection
# ═══════════════════════════════════════════════════════════════════════════


def collect_scatter_points(
    sequences: List[CameraSequence],
    depth_paths: Dict[int, Tuple[Optional[str], Optional[str]]],
    points_3d: torch.Tensor,
    scene_stats: SceneStats,
    device: torch.device,
    focal: float,
    cx: float,
    cy: float,
    flow_magnitude_thresh: float,
    use_consistency_mask: bool,
    subsample_factor: int,
    ransac_iters: int,
    inlier_thresh: float,
    min_anchors: int,
    max_frames_per_cam: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, List[DepthAlignResult]]:
    """Traverse all (camera, frame_pair) and collect 3D scatter points.

    Returns:
        all_positions: ``[N_total, 3]`` world coordinates (CPU).
        all_displacements: ``[N_total, 3]`` world displacements (CPU).
        all_align_results: Alignment diagnostics for each processed frame pair.
    """
    aabb_min = torch.from_numpy(scene_stats.aabb_min.astype(np.float32)).to(device)
    aabb_max = torch.from_numpy(scene_stats.aabb_max.astype(np.float32)).to(device)

    all_pos_list: List[torch.Tensor] = []
    all_disp_list: List[torch.Tensor] = []
    all_align: List[DepthAlignResult] = []
    total_pairs = 0
    valid_pairs = 0

    for seq_idx, seq in enumerate(sequences):
        frames = seq.frames
        n_frames = len(frames)

        # Uniform skip to limit frames
        if n_frames > max_frames_per_cam + 1:
            step = max(1, (n_frames - 1) // max_frames_per_cam)
            indices = list(range(0, n_frames - 1, step))[:max_frames_per_cam]
        else:
            indices = list(range(n_frames - 1))

        cam_pairs = 0
        for idx in indices:
            frame_t = frames[idx]
            frame_tp1 = frames[idx + 1]

            dp_t = depth_paths.get(frame_t.uid, (None, None))[0]
            dp_tp1 = depth_paths.get(frame_tp1.uid, (None, None))[0]

            total_pairs += 1
            result = compute_scene_flow_frame(
                cam_info=frame_t,
                depth_path_t=dp_t,
                depth_path_tp1=dp_tp1,
                points_3d=points_3d,
                scene_stats=scene_stats,
                device=device,
                focal=focal,
                cx=cx,
                cy=cy,
                flow_magnitude_thresh=flow_magnitude_thresh,
                use_consistency_mask=use_consistency_mask,
                subsample_factor=subsample_factor,
                ransac_iters=ransac_iters,
                inlier_thresh=inlier_thresh,
                min_anchors=min_anchors,
            )
            if result is None:
                continue

            pos_w, disp_w, align = result
            all_pos_list.append(pos_w.cpu())
            all_disp_list.append(disp_w.cpu())
            all_align.append(align)
            valid_pairs += 1
            cam_pairs += 1

        logger.info("  Camera %s: %d/%d valid frame pairs.", seq.name, cam_pairs, len(indices))

    if not all_pos_list:
        logger.error("No valid scatter points collected!")
        return torch.zeros(0, 3), torch.zeros(0, 3), all_align

    all_pos = torch.cat(all_pos_list, dim=0)    # [N_total, 3]
    all_disp = torch.cat(all_disp_list, dim=0)  # [N_total, 3]
    logger.info("Collected %d scatter points from %d/%d frame pairs.",
                all_pos.shape[0], valid_pairs, total_pairs)

    return all_pos, all_disp, all_align


def stratified_subsample(
    positions: torch.Tensor,
    displacements: torch.Tensor,
    n_sub: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniformly random subsample scatter points.

    Args:
        positions: ``[N, 3]``
        displacements: ``[N, 3]``
        n_sub: Target number of sub-sampled points.

    Returns:
        ``(positions_sub, displacements_sub)`` each ``[n_sub, 3]``.
    """
    N = positions.shape[0]
    if N <= n_sub:
        return positions, displacements
    idx = torch.randperm(N)[:n_sub]
    return positions[idx], displacements[idx]


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def configure_logging_v2(output_path: str) -> None:
    """Configure logging for v2 analysis."""
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "flow_resolution_analysis_v2.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def parse_args_v2() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate 4-layer teacher HexPlane resolutions (v2: Structure Function).",
    )
    # ── v1 parameters ──
    parser.add_argument("-s", "--source_path", required=True, type=str,
                        help="Path to N3V dataset root.")
    parser.add_argument("-o", "--output_path", type=str,
                        default="output_flow_analysis",
                        help="Directory for log and JSON outputs.")
    parser.add_argument("--num_frames", type=int, default=300,
                        help="Maximum frames per camera to analyze.")
    parser.add_argument("--eval_index", type=int, default=0,
                        help="Held-out test camera index.")
    parser.add_argument("--energy_cutoff", type=float, default=0.95,
                        help="Energy percentile for DFT frequency cutoff.")
    parser.add_argument("--nyquist_margin", type=float, default=2.78,
                        help="Safety multiplier for Nyquist-limited resolution.")
    parser.add_argument("--subsample_factor", type=int, default=4,
                        help="Spatial subsampling step.")
    parser.add_argument("--max_cameras", type=int, default=-1,
                        help="Maximum training cameras to analyze (-1 = all).")
    parser.add_argument("--fixed_res", type=int, default=64,
                        help="Fixed resolution for the first three HexPlane levels.")
    parser.add_argument("--res_candidates", type=str,
                        default="64,96,128,192,256,384,512",
                        help="Comma-separated discrete resolution candidates.")
    parser.add_argument("--flow_magnitude_thresh", type=float, default=0.5,
                        help="Ignore low-magnitude flow vectors below this threshold.")
    parser.add_argument("--use_consistency_mask", action="store_true", default=True)
    parser.add_argument("--disable_consistency_mask", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # ── v2 new parameters ──
    parser.add_argument("--depth_dir_name", type=str, default="depth",
                        help="Depth directory name relative to source_path.")
    parser.add_argument("--sfm_points_path", type=str, default=None,
                        help="Custom SfM PLY path (default: auto-detect).")
    parser.add_argument("--affine_ransac_iters", type=int, default=100,
                        help="RANSAC iterations for depth alignment.")
    parser.add_argument("--affine_inlier_thresh", type=float, default=0.1,
                        help="Absolute residual threshold for RANSAC inliers.")
    parser.add_argument("--min_anchor_points", type=int, default=10,
                        help="Minimum visible SfM anchors for valid alignment.")
    parser.add_argument("--sf_n_sub", type=int, default=10000,
                        help="Number of sub-sampled points for structure function.")
    parser.add_argument("--sf_n_runs", type=int, default=5,
                        help="Number of sub-sampling runs to average.")
    parser.add_argument("--sf_n_bins", type=int, default=64,
                        help="Number of radial bins for structure function.")
    parser.add_argument("--sf_r_min", type=float, default=0.001,
                        help="Minimum separation distance for SF bins.")
    parser.add_argument("--sf_r_max", type=float, default=1.0,
                        help="Maximum separation distance for SF bins.")
    parser.add_argument("--sf_plateau_pct", type=float, default=0.95,
                        help="Plateau fraction for SF cutoff detection.")
    parser.add_argument("--disable_structure_function", action="store_true",
                        help="Disable structure function analysis (DFT-only mode).")
    parser.add_argument("--max_frames_per_cam", type=int, default=50,
                        help="Maximum frame pairs per camera for scatter collection.")

    args = parser.parse_args(sys.argv[1:])
    if args.disable_consistency_mask:
        args.use_consistency_mask = False
    return args


# ═══════════════════════════════════════════════════════════════════════════
# Summary & output
# ═══════════════════════════════════════════════════════════════════════════


def summarize_table_v2(result: Dict) -> str:
    """Format extended summary table with structure function results."""
    lines = [
        "=" * 78,
        "Optical Flow -> HexPlane Resolution Analysis (v2: Structure Function)",
        "=" * 78,
        f"Dataset:       {result['dataset']}",
        f"Cameras:       {result['num_cameras']} train cameras, {result['num_frames']} flow frames/camera",
        f"Image:         {result['image_width']}x{result['image_height']}",
        f"Median focal:  {result['scene']['median_focal']:.2f} px",
        f"Median depth:  {result['scene']['median_depth']:.4f}",
        f"AABB extent:   {np.array2string(np.asarray(result['scene']['aabb_extent']), precision=4)}",
        "-" * 78,
        "TEMPORAL ANALYSIS",
        f"  95% cutoff freq:      {result['analysis']['temporal']['f_t_max']:.4f} cycles/sequence",
        f"  Rt (raw):             {result['analysis']['temporal']['Rt_raw']:.4f}",
        f"  Rt (snapped):         {result['analysis']['temporal']['Rt_snapped']}",
        "-" * 78,
        "SPATIAL ANALYSIS (DFT, baseline)",
        f"  95% cutoff pixel:     {result['analysis']['spatial_dft']['f_s_pixel']:.6f} cycles/pixel",
        f"  95% cutoff canon:     {result['analysis']['spatial_dft']['f_s_canon']:.6f} cycles/canonical",
        f"  Rs_DFT (raw):         {result['analysis']['spatial_dft']['Rs_raw']:.4f}",
        f"  Rs_DFT (snapped):     {result['analysis']['spatial_dft']['Rs_snapped']}",
        "-" * 78,
        "SPATIAL ANALYSIS (gradient ratio, diagnostic)",
        f"  rho (canonical):      {result['analysis']['gradient_ratio']['rho']:.6f}",
        f"  Rs_grad (raw):        {result['analysis']['gradient_ratio']['Rs_raw']:.4f}",
        f"  Rs_grad (snapped):    {result['analysis']['gradient_ratio']['Rs_snapped']}",
    ]

    sf = result["analysis"].get("structure_function")
    if sf is not None:
        lines += [
            "-" * 78,
            "SPATIAL ANALYSIS (Structure Function, primary)",
            f"  Scatter points:       {sf['n_points_total']} total, {sf['n_sub_actual']} sub-sampled",
            f"  D_inf (plateau):      {sf['D_inf']:.6e}",
            f"  D_0 (noise floor):    {sf['D_0']:.6e}",
            f"  r_c (cutoff scale):   {sf['r_c']:.6f}",
            f"  f_c (cutoff freq):    {sf['f_c']:.6f} cycles/canonical",
            f"  Rs_SF (raw):          {sf['Rs_raw']:.4f}",
            f"  Rs_SF (snapped):      {sf['Rs_snapped']}",
        ]
        da = result["analysis"].get("depth_alignment")
        if da is not None:
            lines += [
                "-" * 78,
                "DEPTH ALIGNMENT DIAGNOSTICS",
                f"  Valid frames:         {da['n_valid']}/{da['n_total']}",
                f"  Mean alpha:           {da['mean_alpha']:.4f}",
                f"  Mean beta:            {da['mean_beta']:.4f}",
                f"  Mean inlier RMSE:     {da['mean_inlier_rmse']:.6f}",
                f"  Mean inlier ratio:    {da['mean_inlier_ratio']:.4f}",
            ]

    lines += [
        "=" * 78,
        "RECOMMENDED 4-LAYER HEXPLANE CONFIG",
        f"  hex_spatial_res = \"{result['hex_spatial_res']}\"",
        f"  hex_time_res    = \"{result['hex_time_res']}\"",
        f"  Estimated plane params: {result['estimated_plane_params_m']:.4f} M",
        f"  Primary method:  {result.get('primary_method', 'DFT')}",
        "=" * 78,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════


def run_analysis_v2(args: argparse.Namespace) -> Dict:
    """Complete v2 analysis: v1 DFT baseline + Structure Function primary."""
    source_path = os.path.abspath(args.source_path)
    output_path = os.path.abspath(args.output_path)
    configure_logging_v2(output_path)

    candidates = parse_candidates(args.res_candidates)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Loading training cameras from %s", source_path)

    cam_infos = load_training_cameras(source_path, args.eval_index, args.num_frames)
    sequences = group_sequences(cam_infos, args.max_cameras)
    scene_stats = compute_scene_stats(source_path, sequences[0].frames[0], sequences)

    sample_cam = sequences[0].frames[0]
    focal = fov2focal(sample_cam.FovX, sample_cam.width)
    cx = sample_cam.width / 2.0
    cy = sample_cam.height / 2.0

    # ══════════════════════════════════════════════════════════════════════
    # Branch 1: v1 DFT temporal + spatial analysis (baseline)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Branch 1: v1 DFT temporal + spatial analysis")
    logger.info("=" * 60)

    temporal_cutoffs: List[float] = []
    temporal_vars: List[float] = []
    gradient_ratios: List[Dict[str, float]] = []
    spectrum_sum_total: Optional[torch.Tensor] = None
    spectrum_count_total: Optional[torch.Tensor] = None

    for seq_idx, sequence in enumerate(sequences):
        logger.info("DFT: camera %d/%d: %s (%d frames)",
                    seq_idx + 1, len(sequences), sequence.name, len(sequence.frames))
        flows, masks = stack_sequence(
            sequence, device=device,
            flow_magnitude_thresh=args.flow_magnitude_thresh,
            use_consistency_mask=args.use_consistency_mask,
            subsample_factor=args.subsample_factor,
        )

        t_cutoff, temporal_var = analyze_temporal_cutoff(flows, masks, args.energy_cutoff)
        grad_info = analyze_gradient_ratio(flows, masks, scene_stats, args.num_frames)
        spectrum_sum, spectrum_count = accumulate_spatial_spectrum(
            flows, masks, args.subsample_factor,
        )

        temporal_cutoffs.append(t_cutoff)
        temporal_vars.append(temporal_var)
        gradient_ratios.append(grad_info)

        if spectrum_sum_total is None:
            spectrum_sum_total = spectrum_sum
            spectrum_count_total = spectrum_count
        else:
            spectrum_sum_total += spectrum_sum
            spectrum_count_total += spectrum_count

    # Temporal
    f_t_max = max(temporal_cutoffs) if temporal_cutoffs else 0.0
    Rt_raw = args.nyquist_margin * f_t_max + 1.0
    Rt_snapped = snap_to_candidate(Rt_raw, candidates)

    # Gradient ratio (diagnostic)
    rho_vals = [g["rho"] for g in gradient_ratios]
    rho = float(np.median(rho_vals)) if rho_vals else 1.0
    rms_s_global = float(np.mean([g["rms_spatial_grad"] for g in gradient_ratios])) if gradient_ratios else 0.0
    rms_t_global = float(np.mean([g["rms_temporal_diff"] for g in gradient_ratios])) if gradient_ratios else 0.0
    canon_scale = gradient_ratios[0]["canon_scale"] if gradient_ratios else 1.0
    Rs_ratio_raw = max(1.0, rho * Rt_raw)
    Rs_ratio_snapped = snap_to_candidate(Rs_ratio_raw, candidates)

    # DFT spatial
    if spectrum_sum_total is not None and spectrum_count_total is not None:
        averaged_spectrum = spectrum_sum_total / spectrum_count_total.clamp_min(1.0)
        sh = sample_cam.height // args.subsample_factor + int(sample_cam.height % args.subsample_factor != 0)
        sw = sample_cam.width // args.subsample_factor + int(sample_cam.width % args.subsample_factor != 0)
        _, radial_freqs, _ = build_radial_bins(sh, sw, args.subsample_factor, device)
    else:
        averaged_spectrum = torch.zeros(1, device=device)
        radial_freqs = torch.zeros(1, device=device)

    f_s_pixel = percentile_cutoff(averaged_spectrum, radial_freqs, args.energy_cutoff)
    spatial_extent = float(np.max(scene_stats.extent))
    f_s_canon = f_s_pixel * scene_stats.median_focal * spatial_extent / max(2.0 * scene_stats.median_depth, 1e-6)
    Rs_dft_raw = args.nyquist_margin * f_s_canon + 1.0
    Rs_dft_snapped = snap_to_candidate(Rs_dft_raw, candidates)

    # ══════════════════════════════════════════════════════════════════════
    # Branch 2: Structure Function analysis (primary)
    # ══════════════════════════════════════════════════════════════════════
    sf_result: Optional[Dict] = None
    da_result: Optional[Dict] = None
    use_sf = not args.disable_structure_function

    if use_sf:
        logger.info("=" * 60)
        logger.info("Branch 2: Structure Function analysis")
        logger.info("=" * 60)

        # Step 2a: Discover depth paths
        depth_paths = discover_depth_paths(source_path, cam_infos, args.depth_dir_name)
        n_with_depth = sum(1 for uid, (dp, _) in depth_paths.items() if dp is not None)
        logger.info("Depth maps: %d/%d frames have depth.", n_with_depth, len(depth_paths))

        if n_with_depth < 10:
            logger.warning("Too few depth maps (%d); falling back to DFT-only.", n_with_depth)
            use_sf = False

    if use_sf:
        # Step 2b: Load SfM points
        if args.sfm_points_path:
            ply_path = args.sfm_points_path
        else:
            ply_path = os.path.join(source_path, "points3D_downsample2.ply")
        if not os.path.isfile(ply_path):
            logger.warning("SfM PLY not found at %s; falling back to DFT-only.", ply_path)
            use_sf = False

    if use_sf:
        sfm_points = torch.from_numpy(fetch_ply(ply_path)).to(device)
        logger.info("SfM points: %d points loaded.", sfm_points.shape[0])

        # Step 2c: Collect scatter points
        all_pos, all_disp, all_aligns = collect_scatter_points(
            sequences=sequences,
            depth_paths=depth_paths,
            points_3d=sfm_points,
            scene_stats=scene_stats,
            device=device,
            focal=focal, cx=cx, cy=cy,
            flow_magnitude_thresh=args.flow_magnitude_thresh,
            use_consistency_mask=args.use_consistency_mask,
            subsample_factor=args.subsample_factor,
            ransac_iters=args.affine_ransac_iters,
            inlier_thresh=args.affine_inlier_thresh,
            min_anchors=args.min_anchor_points,
            max_frames_per_cam=args.max_frames_per_cam,
        )

        if all_pos.shape[0] < 100:
            logger.warning("Only %d scatter points; SF unreliable. Falling back to DFT.",
                           all_pos.shape[0])
            use_sf = False

    if use_sf:
        # Transform to canonical
        aabb_min_t = torch.from_numpy(scene_stats.aabb_min.astype(np.float32))
        aabb_max_t = torch.from_numpy(scene_stats.aabb_max.astype(np.float32))
        all_pos_canon, all_disp_canon = to_canonical(all_pos, all_disp, aabb_min_t, aabb_max_t)
        logger.info("Canonical points: %d (after outlier filtering).", all_pos_canon.shape[0])

        # Step 2d: Multi-run structure function
        D_iso_acc = None
        D_x_acc = None
        D_y_acc = None
        D_z_acc = None
        counts_acc = None
        n_sub_actual = min(args.sf_n_sub, all_pos_canon.shape[0])

        for run_i in range(args.sf_n_runs):
            pos_sub, disp_sub = stratified_subsample(
                all_pos_canon, all_disp_canon, args.sf_n_sub,
            )
            D_iso, D_x, D_y, D_z, r_centers, counts = compute_structure_function(
                pos_sub.to(device), disp_sub.to(device),
                n_bins=args.sf_n_bins,
                r_min=args.sf_r_min,
                r_max=args.sf_r_max,
                device=device,
            )
            if D_iso_acc is None:
                D_iso_acc = D_iso.clone()
                D_x_acc = D_x.clone()
                D_y_acc = D_y.clone()
                D_z_acc = D_z.clone()
                counts_acc = counts.clone()
            else:
                D_iso_acc += D_iso
                D_x_acc += D_x
                D_y_acc += D_y
                D_z_acc += D_z
                counts_acc += counts

            logger.info("  SF run %d/%d complete (n_sub=%d).", run_i + 1, args.sf_n_runs, pos_sub.shape[0])

        n_runs = args.sf_n_runs
        D_iso_avg = D_iso_acc / n_runs
        D_x_avg = D_x_acc / n_runs
        D_y_avg = D_y_acc / n_runs
        D_z_avg = D_z_acc / n_runs

        # Step 2e: Plateau and cutoff
        D_inf, D_0, r_c = estimate_plateau_and_cutoff(
            D_iso_avg, r_centers, counts_acc,
            plateau_pct=args.sf_plateau_pct,
        )

        # Step 2f: Resolution conversion
        f_c, Rs_sf_raw, Rs_sf_snapped = cutoff_to_resolution(
            r_c, args.nyquist_margin, candidates,
        )

        logger.info("SF result: D_inf=%.4e, D_0=%.4e, r_c=%.6f, f_c=%.4f, Rs_raw=%.2f, Rs_snap=%d",
                     D_inf, D_0, r_c, f_c, Rs_sf_raw, Rs_sf_snapped)

        sf_result = {
            "D_inf": D_inf,
            "D_0": D_0,
            "r_c": r_c,
            "f_c": f_c,
            "Rs_raw": Rs_sf_raw,
            "Rs_snapped": Rs_sf_snapped,
            "n_points_total": int(all_pos_canon.shape[0]),
            "n_sub_actual": n_sub_actual,
            "n_runs": args.sf_n_runs,
        }

        # Depth alignment diagnostics
        valid_aligns = [a for a in all_aligns if a.valid]
        if valid_aligns:
            da_result = {
                "n_valid": len(valid_aligns),
                "n_total": len(all_aligns),
                "mean_alpha": float(np.mean([a.alpha for a in valid_aligns])),
                "mean_beta": float(np.mean([a.beta for a in valid_aligns])),
                "mean_inlier_rmse": float(np.mean([a.inlier_rmse for a in valid_aligns])),
                "mean_inlier_ratio": float(np.mean([a.inlier_ratio for a in valid_aligns])),
            }

    # ══════════════════════════════════════════════════════════════════════
    # Final recommendation
    # ══════════════════════════════════════════════════════════════════════
    if use_sf and sf_result is not None:
        Rs_primary = max(sf_result["Rs_snapped"], 2 * args.fixed_res)
        primary_method = "StructureFunction"
    else:
        Rs_primary = max(Rs_dft_snapped, 2 * args.fixed_res)
        primary_method = "DFT"

    Rs_primary = snap_to_candidate(float(Rs_primary), candidates)

    fixed_levels = [args.fixed_res] * 3
    spatial_levels = fixed_levels + [Rs_primary]
    time_levels = fixed_levels + [Rt_snapped]

    result: Dict = {
        "dataset": source_path,
        "output_path": output_path,
        "num_cameras": len(sequences),
        "num_frames": min(len(seq.frames) for seq in sequences),
        "image_width": scene_stats.width,
        "image_height": scene_stats.height,
        "hex_spatial_res": ",".join(str(v) for v in spatial_levels),
        "hex_time_res": ",".join(str(v) for v in time_levels),
        "estimated_plane_params_m": estimate_plane_params(spatial_levels, time_levels),
        "primary_method": primary_method,
        "scene": {
            "aabb_min": scene_stats.aabb_min.tolist(),
            "aabb_max": scene_stats.aabb_max.tolist(),
            "aabb_extent": scene_stats.extent.tolist(),
            "scene_center": scene_stats.scene_center.tolist(),
            "median_depth": scene_stats.median_depth,
            "median_focal": scene_stats.median_focal,
        },
        "analysis": {
            "temporal": {
                "f_t_max": f_t_max,
                "Rt_raw": Rt_raw,
                "Rt_snapped": Rt_snapped,
            },
            "gradient_ratio": {
                "rms_spatial_grad": rms_s_global,
                "rms_temporal_diff": rms_t_global,
                "canon_scale": canon_scale,
                "rho": rho,
                "Rs_raw": Rs_ratio_raw,
                "Rs_snapped": Rs_ratio_snapped,
            },
            "spatial_dft": {
                "f_s_pixel": f_s_pixel,
                "f_s_canon": f_s_canon,
                "Rs_raw": Rs_dft_raw,
                "Rs_snapped": Rs_dft_snapped,
            },
        },
        "config": {
            "energy_cutoff": args.energy_cutoff,
            "nyquist_margin": args.nyquist_margin,
            "subsample_factor": args.subsample_factor,
            "fixed_res": args.fixed_res,
            "res_candidates": candidates,
            "flow_magnitude_thresh": args.flow_magnitude_thresh,
            "use_consistency_mask": args.use_consistency_mask,
            "use_structure_function": use_sf,
            "device": str(device),
        },
    }

    if sf_result is not None:
        result["analysis"]["structure_function"] = sf_result
    if da_result is not None:
        result["analysis"]["depth_alignment"] = da_result

    summary = summarize_table_v2(result)
    logger.info("\n%s", summary)

    json_path = os.path.join(output_path, "resolution_config_v2.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Wrote analysis JSON to %s", json_path)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args_v2()
    try:
        run_analysis_v2(args)
    except Exception as exc:
        logger.exception("Flow resolution analysis v2 failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
