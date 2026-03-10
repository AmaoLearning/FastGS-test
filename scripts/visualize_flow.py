"""可视化文件夹中所有 .npy 光流文件，逐张显示。

三面板对比：
  左：原始前向光流
  中：经前后一致性掩码筛选后的前向光流（不可信区域置黑）
  右：经一致性 + 模长阈值筛选后的光流（静态噪声 + 不可信区域置黑）

Usage:
    python scripts/visualize_flow.py --flow_dir "E:\\tmp\\datasets\\N3D\\coffee_martini\\flow\\cam01"
    python scripts/visualize_flow.py --flow_dir "..." --mag_thresh 0.5
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch

# 添加项目根目录到 sys.path，以便 import utils
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.optic_flow_utils import forward_backward_consistency_check


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """将 [H, W, 2] 光流转换为 HSV 色轮可视化 (BGR uint8)。"""
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2 + 180) % 180   # Hue: 方向
    hsv[..., 1] = 255                                      # Saturation: 满
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)  # Value: 模长

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def load_flow(path: str) -> np.ndarray:
    """加载 .npy 光流，统一为 [H, W, 2] 格式。"""
    flow = np.load(path)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)
    return flow


def compute_consistency_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    """计算前-后向一致性掩码，返回 [H, W] bool mask。"""
    fwd_t = torch.from_numpy(flow_fwd.transpose(2, 0, 1)).float()
    bwd_t = torch.from_numpy(flow_bwd.transpose(2, 0, 1)).float()
    mask = forward_backward_consistency_check(fwd_t, bwd_t)  # [1, H, W]
    return mask.squeeze(0).numpy() > 0.5  # [H, W] bool


def compute_magnitude_mask(flow: np.ndarray, thresh: float) -> np.ndarray:
    """计算模长阈值掩码，返回 [H, W] bool mask。|flow| > thresh 的像素为 True。"""
    mag = np.linalg.norm(flow, axis=-1)  # [H, W]
    return mag > thresh


def main():
    parser = argparse.ArgumentParser(description="Visualize .npy optical flow files")
    parser.add_argument("--flow_dir", type=str, required=True,
                        help="Directory containing .npy flow files (of_fwd_*.npy / of_bwd_*.npy)")
    parser.add_argument("--mag_thresh", type=float, default=1.0,
                        help="Magnitude threshold (pixels). Flow below this is treated as static noise. Default: 1.0")
    args = parser.parse_args()

    flow_dir = args.flow_dir
    mag_thresh = args.mag_thresh
    assert os.path.isdir(flow_dir), f"Directory not found: {flow_dir}"

    # 自动配对 fwd / bwd 文件
    all_npy = sorted([f for f in os.listdir(flow_dir) if f.endswith(".npy")])
    fwd_files = sorted([f for f in all_npy if "fwd" in f])
    bwd_files = sorted([f for f in all_npy if "bwd" in f])

    def extract_stem(fname: str) -> str:
        base = os.path.splitext(fname)[0]
        return base.replace("of_fwd_", "").replace("of_bwd_", "")

    fwd_map = {extract_stem(f): f for f in fwd_files}
    bwd_map = {extract_stem(f): f for f in bwd_files}
    paired_stems = sorted(set(fwd_map.keys()) & set(bwd_map.keys()))
    fwd_only_stems = sorted(set(fwd_map.keys()) - set(bwd_map.keys()))
    stems = paired_stems + fwd_only_stems

    if not stems and all_npy:
        stems = [os.path.splitext(f)[0] for f in all_npy]
        fwd_map = {os.path.splitext(f)[0]: f for f in all_npy}
        bwd_map = {}

    assert len(stems) > 0, f"No .npy files found in {flow_dir}"

    n_paired = len(paired_stems)
    print(f"Found {len(fwd_map)} fwd files, {len(bwd_map)} bwd files, {n_paired} paired")
    print(f"Magnitude threshold: {mag_thresh:.2f} px")
    print(f"Total frames to visualize: {len(stems)}")
    print("Controls:  [→ / Space / D] next  |  [← / Backspace / A] prev  |  [Q / Esc] quit")

    idx = 0
    win_name = "Raw | Consistency | Consistency + Magnitude"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        stem = stems[idx]
        fwd_path = os.path.join(flow_dir, fwd_map[stem]) if stem in fwd_map else None
        bwd_path = os.path.join(flow_dir, bwd_map[stem]) if stem in bwd_map else None

        flow_fwd = load_flow(fwd_path)
        vis_raw = flow_to_color(flow_fwd)
        mag_max = np.linalg.norm(flow_fwd, axis=-1).max()

        # ── Panel 2: Consistency Mask Only ──
        if bwd_path is not None:
            flow_bwd = load_flow(bwd_path)
            consist_mask = compute_consistency_mask(flow_fwd, flow_bwd)
            flow_consist = flow_fwd.copy()
            flow_consist[~consist_mask] = 0.0
            vis_consist = flow_to_color(flow_consist)
            consist_ratio = consist_mask.sum() / consist_mask.size * 100
        else:
            vis_consist = np.zeros_like(vis_raw)
            cv2.putText(vis_consist, "No bwd flow", (10, vis_consist.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            consist_mask = None
            consist_ratio = -1

        # ── Panel 3: Consistency + Magnitude ──
        mag_mask = compute_magnitude_mask(flow_fwd, mag_thresh)
        if consist_mask is not None:
            combined_mask = consist_mask & mag_mask
        else:
            combined_mask = mag_mask
        flow_combined = flow_fwd.copy()
        flow_combined[~combined_mask] = 0.0
        vis_combined = flow_to_color(flow_combined)
        combined_ratio = combined_mask.sum() / combined_mask.size * 100
        mag_only_ratio = mag_mask.sum() / mag_mask.size * 100

        # ── Canvas ──
        canvas = np.hstack([vis_raw, vis_consist, vis_combined])
        h_img, w_img = vis_raw.shape[:2]

        # Title row
        cv2.putText(canvas, "Raw Flow", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Consistency Only", (w_img + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Consist + Mag(>{mag_thresh:.2f}px)", (2 * w_img + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Info row
        info_raw = f"[{idx + 1}/{len(stems)}] {stem}  mag_max={mag_max:.2f}px"
        cv2.putText(canvas, info_raw, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


        if consist_ratio >= 0:
            cv2.putText(canvas, f"valid={consist_ratio:.1f}%", (w_img + 10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"valid={combined_ratio:.1f}%  (mag>{mag_thresh}px: {mag_only_ratio:.1f}%)",
                    (2 * w_img + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('d'), 32, 83):
            idx = (idx + 1) % len(stems)
        elif key in (ord('a'), 8, 81):
            idx = (idx - 1) % len(stems)


if __name__ == "__main__":
    main()

    cv2.destroyAllWindows()

