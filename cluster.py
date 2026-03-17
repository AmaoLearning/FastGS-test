"""
Standalone clustering script for trained Gaussian point clouds.

Loads a trained model at a specific iteration, performs dynamic Gaussian
clustering, and renders pseudocolor visualizations.

Usage:
    python cluster.py --model_path /path/to/output --iteration 15000
    python cluster.py -m /path/to/output -i 15000
"""

import argparse
import os
import sys
import torch

from scene import Scene, GaussianModel, DeformModel, DeformModel_4DGS
from utils.cluster_utils import cluster_dynamic_gaussians, render_cluster_pseudocolor
from gaussian_renderer import render_fastgs
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster trained Gaussians")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Iteration to load
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration number to load (default: -1 = latest)")
    
    # Clustering parameters (override defaults)
    parser.add_argument("--dynamic_thresh", type=float, default=None,
                        help="Dynamic probability threshold for clustering")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of KMeans clusters")
    parser.add_argument("--w_xyz", type=float, default=None,
                        help="Weight for 3D position feature")
    parser.add_argument("--w_color", type=float, default=None,
                        help="Weight for color (SH 0th-order) feature")
    parser.add_argument("--w_motion", type=float, default=None,
                        help="Weight for motion (mean deformation) feature")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sigmoid temperature for dynamic prob computation")
    
    # Rendering parameters
    parser.add_argument("--white_background", action="store_true", default=False,
                        help="Use white background")
    
    # Which cameras to render
    parser.add_argument("--render_test", action="store_true", default=True,
                        help="Render test cameras")
    parser.add_argument("--render_train", action="store_true", default=False,
                        help="Render train cameras")
    parser.add_argument("--num_render", type=int, default=1,
                        help="Number of train cameras to render (if --render_train)")
    
    # Output
    parser.add_argument("--save_npz", action="store_true", default=True,
                        help="Save clustering results as .npz")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save pseudocolor images as PNG")
    
    # Get combined args from cfg_args file
    args = get_combined_args(parser)
    
    # Override clustering params from CLI if provided
    if args.dynamic_thresh is None:
        args.dynamic_thresh = getattr(args, 'cluster_dynamic_thresh', 0.5)
    if args.n_clusters is None:
        args.n_clusters = getattr(args, 'cluster_n_clusters', 8)
    if args.w_xyz is None:
        args.w_xyz = getattr(args, 'cluster_w_xyz', 1.0)
    if args.w_color is None:
        args.w_color = getattr(args, 'cluster_w_color', 0.5)
    if args.w_motion is None:
        args.w_motion = getattr(args, 'cluster_w_motion', 0.5)
    
    return args, lp, op, pp


def main():
    args, lp, op, pp = parse_args()
    
    # Extract dataset and pipeline params
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    
    print(f"[INFO] Model: {args.model_path}")
    print(f"[INFO] Iteration: {args.iteration}")
    print(f"[INFO] deform_type: {getattr(dataset, 'deform_type', 'mlp')}")
    print(f"[INFO] is_6dof: {dataset.is_6dof}")
    print(f"[INFO] Clustering params:")
    print(f"    dynamic_thresh: {args.dynamic_thresh}")
    print(f"    n_clusters: {args.n_clusters}")
    print(f"    w_xyz: {args.w_xyz}, w_color: {args.w_color}, w_motion: {args.w_motion}")
    print(f"    temperature: {args.temperature}")
    
    # Create Gaussian model and load from checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Load point cloud from saved iteration
    if args.iteration == -1:
        # Find latest iteration
        pc_dir = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(pc_dir):
            iterations = [d.replace("iteration_", "") for d in os.listdir(pc_dir) 
                        if d.startswith("iteration_") and os.path.isdir(os.path.join(pc_dir, d))]
            if iterations:
                args.iteration = max(int(i) for i in iterations)
                print(f"[INFO] Auto-detected latest iteration: {args.iteration}")
    
    pc_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if os.path.exists(pc_path):
        print(f"[INFO] Loading Gaussians from {pc_path}")
        gaussians.load_ply(pc_path)
    else:
        print(f"[ERROR] Point cloud not found at {pc_path}")
        return
    
    # Create deformation model
    _deform_type = getattr(dataset, "deform_type", "mlp")
    if _deform_type == "4dgs":
        _s_res = tuple(int(x) for x in dataset.hex_spatial_res.split(","))
        _t_res = tuple(int(x) for x in dataset.hex_time_res.split(","))
        deform = DeformModel_4DGS(
            is_blender=dataset.is_blender,
            is_6dof=dataset.is_6dof,
            spatial_resolutions=_s_res,
            time_resolutions=_t_res,
            feat_dim=dataset.hex_feat_dim,
            mlp_hidden_dim=dataset.hex_mlp_hidden,
            mlp_num_hidden=dataset.hex_mlp_layers,
            fusion=dataset.hex_fusion,
        )
    else:
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    
    # Load deform weights
    deform.load_weights(args.model_path, args.iteration)
    
    # Create scene (for camera loading)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration)
    
    # Set up background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Create output directory
    cluster_dir = os.path.join(args.model_path, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────
    # Run clustering
    # ─────────────────────────────────────────────────────────────
    _cluster_save = os.path.join(cluster_dir, f"cluster_iter{args.iteration}.npz")
    
    cluster_result = cluster_dynamic_gaussians(
        gaussians,
        dynamic_thresh=args.dynamic_thresh,
        n_clusters=args.n_clusters,
        w_xyz=args.w_xyz,
        w_color=args.w_color,
        w_motion=args.w_motion,
        temperature=args.temperature,
        iteration=args.iteration,
        save_path=_cluster_save if args.save_npz else None,
    )
    
    gaussians._cluster_labels = cluster_result["labels"]
    
    print(f"[INFO] Clustering complete: {cluster_result['n_dynamic']} dynamic Gaussians")
    print(f"[INFO] Cluster saved to {_cluster_save}")
    
    # ─────────────────────────────────────────────────────────────
    # Render pseudocolor images
    # ─────────────────────────────────────────────────────────────
    
    def render_cameras(cameras, prefix):
        """Render pseudocolor for a list of cameras."""
        for idx, viewpoint_cam in enumerate(cameras):
            scene.ensure_cameras_loaded([viewpoint_cam])
            
            save_path = os.path.join(cluster_dir, f"{prefix}_iter{args.iteration}_{idx:03d}.png")
            
            render_cluster_pseudocolor(
                gaussians,
                labels=cluster_result["labels"],
                viewpoint_cam=viewpoint_cam,
                deform=deform,
                pipe=pipe,
                bg_color=background,
                mult=args.mult,
                is_6dof=dataset.is_6dof,
                save_path=save_path if args.save_images else None,
                iteration=args.iteration,
            )
            
            scene.release_cameras([viewpoint_cam])
            print(f"[INFO] Rendered {save_path}")
    
    # Render test cameras (default)
    if args.render_test:
        test_cams = scene.getTestCameras()
        if test_cams:
            num_cams = min(args.num_render, len(test_cams))
            indices = list(range(0, len(test_cams), len(test_cams) // num_cams))[:num_cams]
            selected_cams = [test_cams[i] for i in indices]
            print(f"[INFO] Rendering {len(selected_cams)} test cameras...")
            render_cameras(selected_cams, "test")
        else:
            print("[WARN] No test cameras found")
    
    # Render train cameras (optional)
    if args.render_train:
        train_cams = scene.getTrainCameras()
        if train_cams:
            # Sample evenly spaced cameras
            num_cams = min(args.num_render, len(train_cams))
            indices = list(range(0, len(train_cams), len(train_cams) // num_cams))[:num_cams]
            selected_cams = [train_cams[i] for i in indices]
            print(f"[INFO] Rendering {len(selected_cams)} train cameras...")
            render_cameras(selected_cams, "train")
        else:
            print("[WARN] No train cameras found")
    
    print(f"[INFO] Clustering and rendering complete!")
    print(f"[INFO] Results saved to: {cluster_dir}")


if __name__ == "__main__":
    main()