"""Generate DA3 depth maps for N3V datasets.

For each camera directory (cam00, cam01, ...) in the N3V scene, runs
Depth Anything 3 inference on the frame sequence and saves per-frame
depth + confidence maps as .npy files.

Output layout:
    <data_dir>/depth/<camXX>/depth_XXXX.npy   # [H, W] float32, meters
    <data_dir>/depth/<camXX>/conf_XXXX.npy    # [H, W] float32, confidence

Usage:
    python scripts/generate_depth_da3.py \
        --data_dir E:/datasets/N3D/coffee_martini \
        --model_name depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --process_res 504 \
        --batch_size 30
"""
import argparse
import os
import sys
import glob

import numpy as np
import torch

# 添加项目根目录到 sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DA3 depth maps for N3V datasets."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to N3V scene root (contains cam*/ directories).",
    )
    parser.add_argument(
        "--model_name", type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="HuggingFace model identifier or local path.",
    )
    parser.add_argument(
        "--process_res", type=int, default=504,
        help="DA3 inference resolution (long-edge upper bound).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=30,
        help="Number of frames per inference batch (controls VRAM).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=300,
        help="Max number of frames per camera to process.",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use fp16 autocast for faster GPU inference (default: True).",
    )
    parser.add_argument(
        "--no_fp16", dest="fp16", action="store_false",
        help="Disable fp16 autocast.",
    )
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, running on CPU — this will be very slow!")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"[DA3] Using GPU: {torch.cuda.get_device_name(device)} "
              f"(cuda:{torch.cuda.current_device()}, "
              f"VRAM {torch.cuda.get_device_properties(device).total_memory // 1024**3} GB)")

    print(f"[DA3] Loading model: {args.model_name}")
    model = DepthAnything3.from_pretrained(args.model_name).to(device)
    model.eval()  # 关闭 BatchNorm/Dropout 的训练模式，避免无效统计计算
    print(f"[DA3] Model ready (fp16={args.fp16 and device.type == 'cuda'})")

    cam_dirs = sorted(glob.glob(os.path.join(args.data_dir, "cam*")))
    if not cam_dirs:
        print(f"[ERROR] No cam* directories found in {args.data_dir}")
        sys.exit(1)

    print(f"[DA3] Found {len(cam_dirs)} cameras, process_res={args.process_res}, "
          f"batch_size={args.batch_size}, num_frames={args.num_frames}")

    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)  # e.g. "cam00"
        image_dir = os.path.join(cam_dir, "images")
        if not os.path.isdir(image_dir):
            print(f"[WARN] Skipping {cam_name}: no images/ subdirectory")
            continue

        frame_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not frame_paths:
            frame_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        frame_paths = frame_paths[:args.num_frames]

        if not frame_paths:
            print(f"[WARN] Skipping {cam_name}: no image files found")
            continue

        depth_out_dir = os.path.join(args.data_dir, "depth", cam_name)
        os.makedirs(depth_out_dir, exist_ok=True)

        # 检查已有输出，跳过已处理的帧
        existing = set(os.listdir(depth_out_dir))
        remaining_paths = []
        for p in frame_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if f"depth_{stem}.npy" not in existing:
                remaining_paths.append(p)

        if not remaining_paths:
            print(f"[DA3] {cam_name}: all {len(frame_paths)} frames already processed, skipping")
            continue

        print(f"[DA3] {cam_name}: processing {len(remaining_paths)}/{len(frame_paths)} frames ...")

        # 分批处理以控制显存
        for start in range(0, len(remaining_paths), args.batch_size):
            batch = remaining_paths[start:start + args.batch_size]

            use_amp = args.fp16 and device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                prediction = model.inference(
                    image=batch,
                    process_res=args.process_res,
                    ref_view_strategy="middle",
                )

            for i, path in enumerate(batch):
                stem = os.path.splitext(os.path.basename(path))[0]
                depth_arr = prediction.depth[i]  # [H, W] float32
                conf_arr = prediction.conf[i]    # [H, W] float32

                np.save(
                    os.path.join(depth_out_dir, f"depth_{stem}.npy"),
                    depth_arr.astype(np.float32),
                )
                np.save(
                    os.path.join(depth_out_dir, f"conf_{stem}.npy"),
                    conf_arr.astype(np.float32),
                )

            batch_end = min(start + args.batch_size, len(remaining_paths))
            print(f"  [{cam_name}] batch {start}-{batch_end}/{len(remaining_paths)} done")

        print(f"[DA3] {cam_name}: {len(remaining_paths)} depth maps saved to {depth_out_dir}")

    print("[DA3] All cameras processed.")


if __name__ == "__main__":
    main()
