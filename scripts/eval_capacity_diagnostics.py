#!/usr/bin/env python3
"""Standalone capacity diagnostics for trained teacher + student deformation models.

Loads pre-trained teacher and student weights from disk, reconstructs the
ClusteredDeformModel, and runs:
  1. SNER (Supra-Nyquist Energy Ratio) — per-cluster HexPlane bandwidth check
  2. MLP effective rank — per-cluster MLP utilisation check

Results are written to:
  - stdout (summary table)
  - ``<output_dir>/capacity_diagnostics.json``
  - (optional) TensorBoard log under ``<output_dir>/tb_diag/``

Usage
-----
::

    python scripts/eval_capacity_diagnostics.py \\
        --model_path  output/my_scene \\
        --teacher_checkpoint_path output/my_scene/deform/iteration_15000/deform.pth \\
        [--iteration -1] \\
        [--n_time_samples 64] \\
        [--output_dir  output/my_scene/diagnostics] \\
        [--write_tensorboard]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional

import torch

# Ensure project root is importable.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Bypass scene/__init__.py (which pulls in dataset_readers → imageio, opencv,
# etc.) by registering a minimal scene package stub before importing the
# submodules we actually need.
import types as _types
if "scene" not in sys.modules:
    _scene_pkg = _types.ModuleType("scene")
    _scene_pkg.__path__ = [os.path.join(_PROJECT_ROOT, "scene")]
    _scene_pkg.__package__ = "scene"
    sys.modules["scene"] = _scene_pkg

from scene.gaussian_model import GaussianModel   # noqa: E402
from scene.deform_model import ClusteredDeformModel  # noqa: E402

from utils.cluster_utils import (
    compute_mlp_effective_rank,
    compute_sner_per_cluster,
    infer_student_configs_from_weights,
)

logger = logging.getLogger("eval_diag")


# ---------------------------------------------------------------------------
# Weight discovery helpers
# ---------------------------------------------------------------------------

_DUAL_RE = re.compile(
    r"deform_cluster_hex(?P<hex_tier>high|medium|low)"
    r"_mlp(?P<mlp_tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
)
_SINGLE_RE = re.compile(
    r"deform_cluster_(?P<tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
)


def _find_latest_iteration(deform_dir: str) -> int:
    """Return the latest iteration number found under *deform_dir*."""
    pattern = re.compile(r"iteration_(\d+)")
    max_iter = -1
    if os.path.isdir(deform_dir):
        for name in os.listdir(deform_dir):
            m = pattern.match(name)
            if m:
                max_iter = max(max_iter, int(m.group(1)))
    return max_iter


def _scan_student_weights(iter_dir: str) -> Dict[int, Dict]:
    """Scan *iter_dir* for student weight files and return tier metadata."""
    cluster_tiers: Dict[int, object] = {}
    for filename in sorted(os.listdir(iter_dir)):
        m = _DUAL_RE.match(filename)
        if m:
            cid = int(m.group("cluster_id"))
            cluster_tiers[cid] = {
                "hex_tier": m.group("hex_tier"),
                "mlp_tier": m.group("mlp_tier"),
            }
            continue
        m = _SINGLE_RE.match(filename)
        if m:
            cid = int(m.group("cluster_id"))
            cluster_tiers[cid] = m.group("tier")
    return cluster_tiers


def _find_point_cloud_ply(model_path: str, iteration: int) -> str:
    """Find the point cloud PLY for a given iteration (or latest)."""
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        raise FileNotFoundError(f"point_cloud directory not found: {pc_dir}")

    if iteration >= 0:
        ply = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud.ply")
        if os.path.isfile(ply):
            return ply

    # Fallback: find latest iteration
    pattern = re.compile(r"iteration_(\d+)")
    max_iter = -1
    for name in os.listdir(pc_dir):
        m = pattern.match(name)
        if m:
            max_iter = max(max_iter, int(m.group(1)))
    if max_iter >= 0:
        ply = os.path.join(pc_dir, f"iteration_{max_iter}", "point_cloud.ply")
        if os.path.isfile(ply):
            return ply
    raise FileNotFoundError(f"No point_cloud.ply found under {pc_dir}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_diagnostics(
    model_path: str,
    teacher_checkpoint_path: str,
    iteration: int = -1,
    n_time_samples: int = 64,
    output_dir: Optional[str] = None,
    write_tensorboard: bool = False,
    capacity_tier_config_path: str = "arguments/capacity_tier_configs.json",
    n_clusters_override: Optional[int] = None,
    max_gaussians_per_cluster: int = 5000,
) -> Dict:
    """Run SNER + MLP effective rank on a trained model.

    Parameters
    ----------
    model_path : str
        Training output directory (contains ``deform/`` and ``point_cloud/``).
    teacher_checkpoint_path : str
        Path to teacher ``deform.pth``.
    iteration : int
        Student weight iteration to load (``-1`` = latest).
    n_time_samples : int
        Number of uniform time steps for SNER analysis.
    output_dir : str or None
        Where to write JSON report; defaults to ``<model_path>/diagnostics``.
    write_tensorboard : bool
        If True, also write TensorBoard scalars.
    capacity_tier_config_path : str
        Path to ``capacity_tier_configs.json``.
    n_clusters_override : int or None
        If set, override auto-detected cluster count.
    max_gaussians_per_cluster : int
        Max Gaussians sampled per cluster for MLP rank analysis.

    Returns
    -------
    dict
        Combined diagnostics result with keys ``"sner"``, ``"mlp_rank"``, ``"meta"``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Discover student weights ──────────────────────────────────────
    deform_dir = os.path.join(model_path, "deform")
    if not os.path.isdir(deform_dir):
        raise FileNotFoundError(f"deform directory not found: {deform_dir}")

    loaded_iter = iteration if iteration >= 0 else _find_latest_iteration(deform_dir)
    if loaded_iter < 0:
        raise FileNotFoundError(f"No iteration directories found in {deform_dir}")

    iter_dir = os.path.join(deform_dir, f"iteration_{loaded_iter}")
    if not os.path.isdir(iter_dir):
        raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")

    cluster_tiers = _scan_student_weights(iter_dir)
    if not cluster_tiers:
        raise FileNotFoundError(f"No student weight files found in {iter_dir}")

    n_clusters = n_clusters_override or (max(cluster_tiers.keys()) + 1)
    logger.info("Detected %d clusters from weight files (iteration %d)", n_clusters, loaded_iter)

    # ── 2. Load capacity tier configs & infer student architectures ──────
    if not os.path.isabs(capacity_tier_config_path):
        capacity_tier_config_path = os.path.join(_PROJECT_ROOT, capacity_tier_config_path)
    if not os.path.isfile(capacity_tier_config_path):
        raise FileNotFoundError(f"Tier config not found: {capacity_tier_config_path}")

    with open(capacity_tier_config_path, "r") as f:
        capacity_tier_configs = json.load(f)

    student_configs = infer_student_configs_from_weights(
        cluster_tiers=cluster_tiers,
        n_clusters=n_clusters,
        capacity_tier_configs=capacity_tier_configs,
    )
    logger.info("Inferred student configs: %s", student_configs)

    # ── 3. Build ClusteredDeformModel ────────────────────────────────────
    clustered_deform = ClusteredDeformModel(
        n_clusters=n_clusters,
        student_configs=student_configs,
    )

    # Load teacher weights
    if not os.path.isfile(teacher_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
    clustered_deform.teacher.load_state_dict(
        torch.load(teacher_checkpoint_path, map_location=device)
    )
    clustered_deform.teacher.eval()
    logger.info("Loaded teacher weights from %s", teacher_checkpoint_path)

    # Load student weights
    clustered_deform.load_weights(model_path, loaded_iter)
    logger.info("Loaded student weights (iteration %d)", loaded_iter)

    # ── 4. Load Gaussians (for xyz + cluster labels) ─────────────────────
    ply_path = _find_point_cloud_ply(model_path, loaded_iter)
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    logger.info("Loaded %d Gaussians from %s", gaussians.get_xyz.shape[0], ply_path)

    cluster_labels = gaussians._cluster_labels
    if cluster_labels is None:
        raise RuntimeError(
            "Loaded point cloud has no cluster_label attribute. "
            "Ensure the PLY was saved after clustering."
        )
    logger.info("Cluster labels range: [%d, %d]", cluster_labels.min().item(), cluster_labels.max().item())

    # Set AABB for normalisation
    clustered_deform.set_aabb(gaussians.get_xyz.detach())

    # ── 5. Run diagnostics ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Capacity Diagnostics — iteration {loaded_iter}")
    print(f"  {n_clusters} clusters, {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"  Teacher: {teacher_checkpoint_path}")
    print(f"{'='*60}\n")

    print("[1/2] Computing SNER (Supra-Nyquist Energy Ratio) ...")
    sner_result = compute_sner_per_cluster(
        gaussians, clustered_deform, cluster_labels,
        n_clusters, n_time_samples=n_time_samples,
    )

    print("[2/2] Computing MLP effective rank ...")
    rank_result = compute_mlp_effective_rank(
        clustered_deform, gaussians, cluster_labels, n_clusters,
        n_time_samples=min(n_time_samples, 16),
        max_gaussians_per_cluster=max_gaussians_per_cluster,
    )

    # ── 6. Print summary table ───────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'Cluster':>8} │ {'SNER':>8} │ {'RMSE':>10} │ {'f_Ny':>6} │ "
          f"{'EffRank':>8} │ {'Util':>6} │ {'HidDim':>6} │ Config")
    print(f"{'─'*80}")
    for k in range(n_clusters):
        cfg = student_configs[k] if k < len(student_configs) else {}
        hex_t = cfg.get("hex_tier", cfg.get("tier", "?"))
        mlp_t = cfg.get("mlp_tier", cfg.get("tier", "?"))
        count = int((cluster_labels == k).sum().item())
        print(
            f"{k:>8} │ {sner_result['sner'][k]:>8.4f} │ "
            f"{sner_result['distill_rmse'][k]:>10.6f} │ "
            f"{sner_result['nyquist_freq'][k]:>6.0f} │ "
            f"{rank_result['effective_rank'][k]:>8.2f} │ "
            f"{rank_result['utilisation'][k]:>6.3f} │ "
            f"{rank_result['hidden_dim'][k]:>6} │ "
            f"hex={hex_t} mlp={mlp_t} (N={count})"
        )
    print(f"{'─'*80}\n")

    # Warnings
    for k in range(n_clusters):
        sner_val = sner_result["sner"][k]
        rmse_val = sner_result["distill_rmse"][k]
        util_val = rank_result["utilisation"][k]
        if sner_val > 0.15 and rmse_val > 0.01:
            print(f"  ⚠ Cluster {k}: SNER={sner_val:.4f}, RMSE={rmse_val:.6f} "
                  "→ consider increasing HexPlane time resolution")
        if util_val > 0.9:
            print(f"  ⚠ Cluster {k}: MLP utilisation={util_val:.3f} "
                  "→ MLP near saturation, consider increasing mlp_hidden_dim")
        if util_val < 0.2 and rank_result["hidden_dim"][k] > 0:
            print(f"  ⚠ Cluster {k}: MLP utilisation={util_val:.3f} "
                  "→ MLP heavily over-provisioned, consider decreasing mlp_hidden_dim")

    # ── 7. Build result dict ─────────────────────────────────────────────
    result = {
        "meta": {
            "model_path": model_path,
            "teacher_checkpoint": teacher_checkpoint_path,
            "iteration": loaded_iter,
            "n_clusters": n_clusters,
            "n_gaussians": int(gaussians.get_xyz.shape[0]),
            "n_time_samples": n_time_samples,
        },
        "sner": sner_result,
        "mlp_rank": rank_result,
        "student_configs": [
            {k: v for k, v in cfg.items() if not k.startswith("_")}
            for cfg in student_configs
        ],
    }

    # ── 8. Save JSON report ──────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(model_path, "diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "capacity_diagnostics.json")

    # Convert non-serialisable types
    def _to_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (tuple,)):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_to_json)
    print(f"Report saved to {json_path}")

    # ── 9. Optional TensorBoard ──────────────────────────────────────────
    if write_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(output_dir, "tb_diag")
            writer = SummaryWriter(log_dir=tb_dir)
            for k in range(n_clusters):
                writer.add_scalar(f"capacity_diag/sner_{k}",
                                  sner_result["sner"][k], loaded_iter)
                writer.add_scalar(f"capacity_diag/distill_rmse_{k}",
                                  sner_result["distill_rmse"][k], loaded_iter)
                writer.add_scalar(f"capacity_diag/mlp_eff_rank_{k}",
                                  rank_result["effective_rank"][k], loaded_iter)
                writer.add_scalar(f"capacity_diag/mlp_utilisation_{k}",
                                  rank_result["utilisation"][k], loaded_iter)
            writer.close()
            print(f"TensorBoard logs written to {tb_dir}")
        except ImportError:
            print("[WARNING] tensorboard not installed, skipping TB output.")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone capacity diagnostics for trained teacher + student deformation models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Training output directory (contains deform/ and point_cloud/).",
    )
    parser.add_argument(
        "--teacher_checkpoint_path", type=str, required=True,
        help="Path to teacher deform.pth.",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Student weight iteration to evaluate (-1 = latest).",
    )
    parser.add_argument(
        "--n_time_samples", type=int, default=64,
        help="Uniform time steps for SNER analysis.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for reports (default: <model_path>/diagnostics).",
    )
    parser.add_argument(
        "--write_tensorboard", action="store_true",
        help="Also write TensorBoard scalar logs.",
    )
    parser.add_argument(
        "--capacity_tier_config_path", type=str,
        default="arguments/capacity_tier_configs.json",
        help="Path to capacity_tier_configs.json.",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=None,
        help="Override auto-detected cluster count.",
    )
    parser.add_argument(
        "--max_gaussians_per_cluster", type=int, default=5000,
        help="Max Gaussians sampled per cluster for MLP rank SVD.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_diagnostics(
        model_path=args.model_path,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        iteration=args.iteration,
        n_time_samples=args.n_time_samples,
        output_dir=args.output_dir,
        write_tensorboard=args.write_tensorboard,
        capacity_tier_config_path=args.capacity_tier_config_path,
        n_clusters_override=args.n_clusters,
        max_gaussians_per_cluster=args.max_gaussians_per_cluster,
    )


if __name__ == "__main__":
    main()
