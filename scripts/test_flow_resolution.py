#!/usr/bin/env python3
"""Estimate teacher HexPlane spatio-temporal resolutions from multi-view optical flow.

The script is designed for N3V / Neur3D style datasets with precomputed optical
flow under ``optical_flow/<cam_name>/of_fwd_XXXX.npy``. It keeps the first three
HexPlane levels fixed to 64 and allocates the last level from flow statistics.

Outputs:
  1. stdout summary table
  2. ``<output_path>/flow_resolution_analysis.log``
  3. ``<output_path>/resolution_config.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

from plyfile import PlyData


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


logger = logging.getLogger("flow_resolution")


def forward_backward_consistency_check(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    alpha1: float = 0.005,
    alpha2: float = 0.2,
) -> torch.Tensor:
    squeeze = False
    if flow_fwd.dim() == 3:
        flow_fwd = flow_fwd.unsqueeze(0)
        flow_bwd = flow_bwd.unsqueeze(0)
        squeeze = True

    batch_size, _, height, width = flow_fwd.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=flow_fwd.device),
        torch.arange(width, dtype=torch.float32, device=flow_fwd.device),
        indexing="ij",
    )

    sample_x = grid_x.unsqueeze(0) + flow_fwd[:, 0]
    sample_y = grid_y.unsqueeze(0) + flow_fwd[:, 1]
    out_of_bounds = (
        (sample_x < 0) | (sample_x > width - 1) |
        (sample_y < 0) | (sample_y > height - 1)
    )

    sample_x_norm = 2.0 * sample_x / max(width - 1, 1) - 1.0
    sample_y_norm = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)

    flow_bwd_warped = F.grid_sample(
        flow_bwd,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    cycle_error = torch.norm(flow_fwd + flow_bwd_warped, dim=1, keepdim=True)
    mag_fwd = torch.norm(flow_fwd, dim=1, keepdim=True)
    mag_bwd = torch.norm(flow_bwd_warped, dim=1, keepdim=True)
    threshold = alpha1 * (mag_fwd + mag_bwd) + alpha2

    valid_mask = (cycle_error < threshold).float()
    valid_mask = valid_mask.masked_fill(out_of_bounds.unsqueeze(1), 0.0)
    if squeeze:
        valid_mask = valid_mask.squeeze(0)
    return valid_mask


@dataclass
class CameraSequence:
    name: str
    frames: List[object]


class LazyCameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    flow_fwd_path: Optional[str] = None
    flow_bwd_path: Optional[str] = None


@dataclass
class SceneStats:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    extent: np.ndarray
    scene_center: np.ndarray
    median_depth: float
    median_focal: float
    width: int
    height: int


def read_n3v_cameras_lazy(
    path: str,
    split: str,
    eval_index: int = 0,
    num_frames: int = 300,
    load_flow: bool = False,
) -> List[LazyCameraInfo]:
    poses_path = os.path.join(path, "poses_bounds.npy")
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"poses_bounds.npy not found under {path}")

    poses_arr = np.load(poses_path)
    poses = poses_arr[:, :-2].reshape(-1, 3, 5)

    _, _, focal_raw = poses[0, :, -1]
    img_w, img_h = 1352, 1014
    downsample_factor = 2704.0 / img_w
    focal = focal_raw / downsample_factor
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], axis=-1)

    videos = sorted(glob(os.path.join(path, "cam*")))
    if not videos:
        raise FileNotFoundError(f"No cam* directories found under {path}")
    if len(videos) != poses.shape[0]:
        raise RuntimeError(
            f"Camera count mismatch: {len(videos)} dirs vs {poses.shape[0]} poses"
        )

    fov_x = focal2fov(focal, img_w)
    fov_y = focal2fov(focal, img_h)
    flow_root = os.path.join(path, "optical_flow")
    has_flow = load_flow and os.path.isdir(flow_root)

    cam_infos: List[LazyCameraInfo] = []
    global_idx = 0
    for cam_i, video_dir in enumerate(videos):
        if split == "train" and cam_i == eval_index:
            continue
        if split == "test" and cam_i != eval_index:
            continue

        pose = np.array(poses[cam_i])
        R = pose[:3, :3].copy()
        R = -R
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)

        cam_name = os.path.basename(video_dir)
        image_dir = os.path.join(video_dir, "images")
        image_names = sorted(os.listdir(image_dir))

        for frame_idx, image_name in enumerate(image_names[:num_frames]):
            image_path = os.path.join(image_dir, image_name)
            fid = frame_idx / num_frames
            flow_fwd_path = None
            flow_bwd_path = None

            if has_flow:
                stem = Path(image_name).stem
                flow_cam_dir = os.path.join(flow_root, cam_name)
                fwd_path = os.path.join(flow_cam_dir, f"of_fwd_{stem}.npy")
                bwd_path = os.path.join(flow_cam_dir, f"of_bwd_{stem}.npy")
                if os.path.isfile(fwd_path):
                    flow_fwd_path = fwd_path
                if os.path.isfile(bwd_path):
                    flow_bwd_path = bwd_path

            cam_infos.append(LazyCameraInfo(
                uid=global_idx,
                R=R,
                T=T,
                FovY=fov_y,
                FovX=fov_x,
                image_path=image_path,
                image_name=str(global_idx),
                width=img_w,
                height=img_h,
                fid=fid,
                flow_fwd_path=flow_fwd_path,
                flow_bwd_path=flow_bwd_path,
            ))
            global_idx += 1

    return cam_infos


def fetch_ply(path: str) -> np.ndarray:
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    return np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate 4-layer teacher HexPlane resolutions from optical flow.",
    )
    parser.add_argument("-s", "--source_path", required=True, type=str,
                        help="Path to N3V dataset root.")
    parser.add_argument("-o", "--output_path", type=str,
                        default="output_flow_analysis",
                        help="Directory for log and JSON outputs.")
    parser.add_argument("--num_frames", type=int, default=300,
                        help="Maximum frames per camera to analyze.")
    parser.add_argument("--eval_index", type=int, default=0,
                        help="Held-out test camera index, matching train.py split.")
    parser.add_argument("--energy_cutoff", type=float, default=0.95,
                        help="Energy percentile used for frequency cutoff.")
    parser.add_argument("--nyquist_margin", type=float, default=2.78,
                        help="Safety multiplier for Nyquist-limited resolution.")
    parser.add_argument("--subsample_factor", type=int, default=4,
                        help="Spatial subsampling step for flow analysis.")
    parser.add_argument("--max_cameras", type=int, default=-1,
                        help="Maximum number of training cameras to analyze (-1 = all).")
    parser.add_argument("--fixed_res", type=int, default=64,
                        help="Fixed resolution for the first three HexPlane levels.")
    parser.add_argument("--res_candidates", type=str,
                        default="64,96,128,192,256,384,512",
                        help="Comma-separated discrete resolution candidates.")
    parser.add_argument("--flow_magnitude_thresh", type=float, default=0.5,
                        help="Ignore low-magnitude flow vectors below this threshold.")
    parser.add_argument("--use_consistency_mask", action="store_true", default=True,
                        help="Use forward-backward consistency masks when available.")
    parser.add_argument("--disable_consistency_mask", action="store_true",
                        help="Disable forward-backward consistency masks.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device, e.g. cuda or cpu.")
    args = parser.parse_args(sys.argv[1:])
    if args.disable_consistency_mask:
        args.use_consistency_mask = False
    return args


def configure_logging(output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "flow_resolution_analysis.log")

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


def parse_candidates(raw: str) -> List[int]:
    values = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("res_candidates must contain at least one integer.")
    return sorted(set(values))


def load_training_cameras(source_path: str, eval_index: int, num_frames: int) -> List[object]:
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Dataset directory not found: {source_path}")

    cam_infos = read_n3v_cameras_lazy(
        source_path,
        split="train",
        eval_index=eval_index,
        num_frames=num_frames,
        load_flow=True,
    )
    if not cam_infos:
        raise RuntimeError("No training cameras found for the provided dataset.")

    flow_count = sum(1 for cam in cam_infos if cam.flow_fwd_path is not None)
    if flow_count == 0:
        raise FileNotFoundError(
            "No forward optical flow files found. Expected optical_flow/<cam>/of_fwd_XXXX.npy."
        )
    return cam_infos


def group_sequences(cam_infos: Sequence[object], max_cameras: int) -> List[CameraSequence]:
    grouped: Dict[str, List[object]] = {}
    for cam in cam_infos:
        if cam.flow_fwd_path is None:
            continue
        cam_name = Path(cam.image_path).parent.parent.name
        grouped.setdefault(cam_name, []).append(cam)

    sequences: List[CameraSequence] = []
    for cam_name in sorted(grouped.keys()):
        frames = sorted(grouped[cam_name], key=lambda item: (item.fid, item.image_path))
        sequences.append(CameraSequence(name=cam_name, frames=frames))

    if max_cameras > 0:
        sequences = sequences[:max_cameras]
    if not sequences:
        raise RuntimeError("No camera sequences with forward flow files were available.")
    return sequences


def compute_scene_stats(source_path: str, sample_cam: object, sequences: Sequence[CameraSequence]) -> SceneStats:
    ply_path = os.path.join(source_path, "points3D_downsample2.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    points = fetch_ply(ply_path)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = np.clip(pmax - pmin, 1e-6, None)
    aabb_min = pmin - 0.1 * extent
    aabb_max = pmax + 0.1 * extent
    padded_extent = aabb_max - aabb_min
    scene_center = 0.5 * (aabb_min + aabb_max)

    focals: List[float] = []
    depths: List[float] = []
    for seq in sequences:
        frame0 = seq.frames[0]
        focals.append(float(fov2focal(frame0.FovX, frame0.width)))
        cam_center = -np.asarray(frame0.R, dtype=np.float32) @ np.asarray(frame0.T, dtype=np.float32)
        depths.append(float(np.linalg.norm(cam_center - scene_center)))

    return SceneStats(
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        extent=padded_extent,
        scene_center=scene_center,
        median_depth=float(np.median(depths)),
        median_focal=float(np.median(focals)),
        width=int(sample_cam.width),
        height=int(sample_cam.height),
    )


def load_flow_with_mask(
    cam_info: object,
    device: torch.device,
    flow_magnitude_thresh: float,
    use_consistency_mask: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cam_info.flow_fwd_path is None:
        raise FileNotFoundError(f"Missing forward flow for {cam_info.image_path}")

    flow_fwd_np = np.load(cam_info.flow_fwd_path)
    flow_fwd = torch.from_numpy(flow_fwd_np).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    mask = torch.ones((1, flow_fwd.shape[1], flow_fwd.shape[2]), device=device, dtype=torch.float32)

    if use_consistency_mask and cam_info.flow_bwd_path is not None and os.path.isfile(cam_info.flow_bwd_path):
        flow_bwd_np = np.load(cam_info.flow_bwd_path)
        flow_bwd = torch.from_numpy(flow_bwd_np).permute(2, 0, 1).to(device=device, dtype=torch.float32)
        mask = forward_backward_consistency_check(flow_fwd, flow_bwd).to(dtype=torch.float32)

    if flow_magnitude_thresh > 0:
        magnitude = torch.norm(flow_fwd, dim=0, keepdim=True)
        mask = mask * (magnitude > flow_magnitude_thresh).to(dtype=torch.float32)

    return flow_fwd, mask


def subsample_sequence(
    flow: torch.Tensor,
    mask: torch.Tensor,
    subsample_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        flow[:, ::subsample_factor, ::subsample_factor],
        mask[:, ::subsample_factor, ::subsample_factor],
    )


def stack_sequence(
    sequence: CameraSequence,
    device: torch.device,
    flow_magnitude_thresh: float,
    use_consistency_mask: bool,
    subsample_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flow_frames: List[torch.Tensor] = []
    mask_frames: List[torch.Tensor] = []
    for cam_info in sequence.frames:
        flow, mask = load_flow_with_mask(
            cam_info,
            device=device,
            flow_magnitude_thresh=flow_magnitude_thresh,
            use_consistency_mask=use_consistency_mask,
        )
        flow, mask = subsample_sequence(flow, mask, subsample_factor)
        flow_frames.append(flow)
        mask_frames.append(mask)

    flows = torch.stack(flow_frames, dim=0)
    masks = torch.stack(mask_frames, dim=0)
    return flows, masks


def masked_temporal_fill(flows: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    data = flows.permute(1, 2, 0, 3).reshape(-1, flows.shape[0], flows.shape[1])
    mask = masks.permute(1, 2, 0, 3).reshape(-1, masks.shape[0], 1)

    valid_ratio = mask.squeeze(-1).mean(dim=1)
    keep = valid_ratio >= 0.5
    if int(keep.sum().item()) == 0:
        keep = torch.ones_like(valid_ratio, dtype=torch.bool)

    data = data[keep]
    mask = mask[keep]
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (data * mask).sum(dim=1, keepdim=True) / counts
    filled = torch.where(mask.bool(), data, mean)
    return filled, mask


def percentile_cutoff(power: torch.Tensor, freqs: torch.Tensor, energy_cutoff: float) -> float:
    if power.numel() == 0 or float(power.sum().item()) <= 0:
        return 0.0

    power = power[1:]
    freqs = freqs[1:]
    if power.numel() == 0 or float(power.sum().item()) <= 0:
        return 0.0

    cumulative = torch.cumsum(power, dim=0) / power.sum()
    index = int(torch.searchsorted(cumulative, torch.tensor(energy_cutoff, device=power.device)).item())
    index = min(index, freqs.numel() - 1)
    return float(freqs[index].item())


def analyze_temporal_cutoff(
    flows: torch.Tensor,
    masks: torch.Tensor,
    energy_cutoff: float,
) -> Tuple[float, float]:
    filled, mask = masked_temporal_fill(flows, masks)
    if filled.shape[1] < 2:
        return 0.0, 0.0

    centered = filled - filled.mean(dim=1, keepdim=True)
    fft = torch.fft.rfft(centered, dim=1)
    power = (fft.abs() ** 2).mean(dim=(0, 2))
    freqs = torch.fft.rfftfreq(filled.shape[1], d=1.0 / float(filled.shape[1])).to(power.device)

    counts = mask.sum(dim=1).clamp_min(1.0)
    mean = (filled * mask).sum(dim=1) / counts
    temporal_var = (((filled - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / counts
    temporal_var = float(temporal_var.mean().item())

    return percentile_cutoff(power, freqs, energy_cutoff), temporal_var


def analyze_spatial_variance(flows: torch.Tensor, masks: torch.Tensor) -> float:
    variances: List[float] = []
    for frame_idx in range(flows.shape[0]):
        flow = flows[frame_idx]
        mask = masks[frame_idx, 0] > 0.5
        if int(mask.sum().item()) < 4:
            continue
        valid = flow[:, mask]
        variances.append(float(valid.var(dim=1, unbiased=False).mean().item()))
    return float(np.mean(variances)) if variances else 0.0


def build_radial_bins(height: int, width: int, subsample_factor: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    fy = torch.fft.fftfreq(height, d=float(subsample_factor), device=device)
    fx = torch.fft.rfftfreq(width, d=float(subsample_factor), device=device)
    rr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    n_bins = max(8, min(128, max(height, width // 2 + 1)))
    edges = torch.linspace(0.0, float(rr.max().item()) + 1e-6, n_bins + 1, device=device)
    flat_rr = rr.reshape(-1)
    bin_idx = torch.bucketize(flat_rr, edges, right=False) - 1
    bin_idx = torch.clamp(bin_idx, min=0, max=n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_idx, centers, n_bins


def accumulate_spatial_spectrum(
    flows: torch.Tensor,
    masks: torch.Tensor,
    subsample_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = flows.device
    height = flows.shape[2]
    width = flows.shape[3]
    bin_idx, centers, n_bins = build_radial_bins(height, width, subsample_factor, device)
    spectrum_sum = torch.zeros(n_bins, device=device)
    spectrum_count = torch.zeros(n_bins, device=device)

    for frame_idx in range(flows.shape[0]):
        flow = flows[frame_idx]
        mask = masks[frame_idx]
        valid_count = mask.sum()
        if float(valid_count.item()) < 4:
            continue

        comp_means = (flow * mask).sum(dim=(1, 2), keepdim=True) / valid_count.clamp_min(1.0)
        centered = (flow - comp_means) * mask
        spec = torch.fft.rfft2(centered, dim=(-2, -1))
        power = (spec.abs() ** 2).mean(dim=0).reshape(-1)

        spectrum_sum += torch.bincount(bin_idx, weights=power, minlength=n_bins)
        spectrum_count += torch.bincount(bin_idx, minlength=n_bins).to(dtype=torch.float32)

    return spectrum_sum, spectrum_count


def snap_to_candidate(value: float, candidates: Sequence[int]) -> int:
    target = int(math.ceil(value))
    for candidate in candidates:
        if candidate >= target:
            return candidate
    logger.warning("Requested resolution %.2f exceeds max candidate %d; clamping.", value, candidates[-1])
    return int(candidates[-1])


def estimate_plane_params(
    spatial_resolutions: Sequence[int],
    time_resolutions: Sequence[int],
    feat_dim: int = 16,
) -> float:
    params = 0
    for s_res, t_res in zip(spatial_resolutions, time_resolutions):
        params += feat_dim * (3 * s_res * s_res + 3 * s_res * t_res)
    return params / 1e6


def summarize_table(result: Dict) -> str:
    lines = [
        "=" * 72,
        "Optical Flow -> HexPlane Resolution Analysis",
        "=" * 72,
        f"Dataset:       {result['dataset']}",
        f"Cameras:       {result['num_cameras']} train cameras, {result['num_frames']} flow frames/camera",
        f"Image:         {result['image_width']}x{result['image_height']}",
        f"Median focal:  {result['scene']['median_focal']:.2f} px",
        f"Median depth:  {result['scene']['median_depth']:.4f}",
        f"AABB extent:   {np.array2string(np.asarray(result['scene']['aabb_extent']), precision=4)}",
        "-" * 72,
        "TEMPORAL ANALYSIS",
        f"  95% cutoff freq:      {result['analysis']['temporal']['f_t_max']:.4f} cycles/sequence",
        f"  Rt (raw):             {result['analysis']['temporal']['Rt_raw']:.4f}",
        f"  Rt (snapped):         {result['analysis']['temporal']['Rt_snapped']}",
        "-" * 72,
        "SPATIAL ANALYSIS (ratio method)",
        f"  Spatial variance:     {result['analysis']['spatial_ratio']['spatial_var']:.6f}",
        f"  Temporal variance:    {result['analysis']['spatial_ratio']['temporal_var']:.6f}",
        f"  rho = Var_s / Var_t:  {result['analysis']['spatial_ratio']['rho']:.6f}",
        f"  Rs (raw):             {result['analysis']['spatial_ratio']['Rs_raw']:.4f}",
        f"  Rs (snapped):         {result['analysis']['spatial_ratio']['Rs_snapped']}",
        "-" * 72,
        "SPATIAL ANALYSIS (DFT reference)",
        f"  95% cutoff pixel:     {result['analysis']['spatial_dft']['f_s_pixel']:.6f} cycles/pixel",
        f"  95% cutoff canon:     {result['analysis']['spatial_dft']['f_s_canon']:.6f} cycles/canonical",
        f"  Rs_DFT (raw):         {result['analysis']['spatial_dft']['Rs_raw']:.4f}",
        f"  Rs_DFT (snapped):     {result['analysis']['spatial_dft']['Rs_snapped']}",
        "=" * 72,
        "RECOMMENDED 4-LAYER HEXPLANE CONFIG",
        f"  hex_spatial_res = \"{result['hex_spatial_res']}\"",
        f"  hex_time_res    = \"{result['hex_time_res']}\"",
        f"  Estimated plane params: {result['estimated_plane_params_m']:.4f} M",
        "=" * 72,
    ]
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> Dict:
    source_path = os.path.abspath(args.source_path)
    output_path = os.path.abspath(args.output_path)
    configure_logging(output_path)

    candidates = parse_candidates(args.res_candidates)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Loading training cameras from %s", source_path)

    cam_infos = load_training_cameras(source_path, args.eval_index, args.num_frames)
    sequences = group_sequences(cam_infos, args.max_cameras)
    scene_stats = compute_scene_stats(source_path, sequences[0].frames[0], sequences)

    temporal_cutoffs: List[float] = []
    temporal_vars: List[float] = []
    spatial_vars: List[float] = []
    spectrum_sum_total: Optional[torch.Tensor] = None
    spectrum_count_total: Optional[torch.Tensor] = None

    for seq_idx, sequence in enumerate(sequences):
        logger.info("Analyzing camera %d/%d: %s (%d frames)",
                    seq_idx + 1, len(sequences), sequence.name, len(sequence.frames))
        flows, masks = stack_sequence(
            sequence,
            device=device,
            flow_magnitude_thresh=args.flow_magnitude_thresh,
            use_consistency_mask=args.use_consistency_mask,
            subsample_factor=args.subsample_factor,
        )

        t_cutoff, temporal_var = analyze_temporal_cutoff(
            flows,
            masks,
            energy_cutoff=args.energy_cutoff,
        )
        spatial_var = analyze_spatial_variance(flows, masks)
        spectrum_sum, spectrum_count = accumulate_spatial_spectrum(
            flows,
            masks,
            subsample_factor=args.subsample_factor,
        )

        temporal_cutoffs.append(t_cutoff)
        temporal_vars.append(temporal_var)
        spatial_vars.append(spatial_var)

        if spectrum_sum_total is None:
            spectrum_sum_total = spectrum_sum
            spectrum_count_total = spectrum_count
        else:
            spectrum_sum_total = spectrum_sum_total + spectrum_sum
            spectrum_count_total = spectrum_count_total + spectrum_count

    f_t_max = max(temporal_cutoffs) if temporal_cutoffs else 0.0
    Rt_raw = args.nyquist_margin * f_t_max + 1.0
    Rt_snapped = snap_to_candidate(Rt_raw, candidates)

    spatial_var_global = float(np.mean(spatial_vars)) if spatial_vars else 0.0
    temporal_var_global = float(np.mean(temporal_vars)) if temporal_vars else 0.0
    if temporal_var_global <= 1e-8:
        logger.warning("Temporal variance is near zero; forcing rho=1.0 to avoid instability.")
        rho = 1.0
    else:
        rho = spatial_var_global / temporal_var_global

    Rs_ratio_raw = max(1.0, rho * Rt_raw)
    Rs_ratio_snapped = snap_to_candidate(Rs_ratio_raw, candidates)

    if spectrum_sum_total is None or spectrum_count_total is None:
        averaged_spectrum = torch.zeros(1, device=device)
        radial_freqs = torch.zeros(1, device=device)
    else:
        averaged_spectrum = spectrum_sum_total / spectrum_count_total.clamp_min(1.0)
        _, radial_freqs, _ = build_radial_bins(
            height=sequences[0].frames[0].height // args.subsample_factor + int(sequences[0].frames[0].height % args.subsample_factor != 0),
            width=sequences[0].frames[0].width // args.subsample_factor + int(sequences[0].frames[0].width % args.subsample_factor != 0),
            subsample_factor=args.subsample_factor,
            device=device,
        )

    f_s_pixel = percentile_cutoff(averaged_spectrum, radial_freqs, args.energy_cutoff)
    spatial_extent = float(np.max(scene_stats.extent))
    f_s_canon = f_s_pixel * scene_stats.median_focal * spatial_extent / max(2.0 * scene_stats.median_depth, 1e-6)
    Rs_dft_raw = args.nyquist_margin * f_s_canon + 1.0
    Rs_dft_snapped = snap_to_candidate(Rs_dft_raw, candidates)

    fixed_levels = [args.fixed_res, args.fixed_res, args.fixed_res]
    spatial_levels = fixed_levels + [Rs_ratio_snapped]
    time_levels = fixed_levels + [Rt_snapped]

    result = {
        "dataset": source_path,
        "output_path": output_path,
        "num_cameras": len(sequences),
        "num_frames": min(len(seq.frames) for seq in sequences),
        "image_width": scene_stats.width,
        "image_height": scene_stats.height,
        "hex_spatial_res": ",".join(str(v) for v in spatial_levels),
        "hex_time_res": ",".join(str(v) for v in time_levels),
        "estimated_plane_params_m": estimate_plane_params(spatial_levels, time_levels),
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
            "spatial_ratio": {
                "spatial_var": spatial_var_global,
                "temporal_var": temporal_var_global,
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
            "device": str(device),
        },
    }

    summary = summarize_table(result)
    logger.info("\n%s", summary)

    json_path = os.path.join(output_path, "resolution_config.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Wrote analysis JSON to %s", json_path)

    return result


def main() -> None:
    args = parse_args()
    try:
        run_analysis(args)
    except Exception as exc:
        logger.exception("Flow resolution analysis failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
