#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel, DeformModel_4DGS, ClusteredDeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time
import re
import glob
import json
from utils.cluster_utils import infer_student_configs_from_weights


def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, args, deform, use_dynamic_sep=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    total_time = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()

        # LazyCamera: image starts as None; load from disk on demand
        if view.original_image is None and hasattr(view, 'load_image_to_gpu'):
            view.load_image_to_gpu('cuda')

        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        # ---- timing starts (deform + render only, excludes I/O) ----
        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)

        torch.cuda.synchronize()
        total_time += time.time() - t_start
        # ---- timing ends ----

        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        # Free image VRAM — thousands of cameras cannot all stay resident
        if hasattr(view, 'unload_image'):
            view.unload_image()

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")


@torch.no_grad()
def _build_render_flow_cache(
    gaussians,
    deform,
    num_frames: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Pre-populate ``gaussians._deform_cache`` for flow-mode rendering.

    Runs a canonical-routing forward pass (``prev_xyz=None``) over all T
    discrete time steps using the **final trained students**.  The resulting
    cache is then used as ``prev_xyz`` in the subsequent render pass so that
    flow-mode cluster routing at render time matches the training behaviour.

    Parameters
    ----------
    gaussians : GaussianModel
    deform    : ClusteredDeformModel
    num_frames: int  – total number of discrete time steps T
    dtype     : storage dtype for the cache (float16 saves RAM)
    """
    time_inputs = torch.linspace(0.0, 1.0, num_frames, device="cuda")

    def _student_canonical_fn(xyz: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # canonical routing: prev_xyz=None → hard argmin on canonical positions
        d_xyz, _, _ = deform.step(
            xyz, time_emb, gaussians._cluster_labels, prev_xyz=None
        )
        return d_xyz

    gaussians.init_deform_cache(
        total_frames=num_frames,
        dtype=dtype,
        teacher_fn=_student_canonical_fn,
        time_inputs=time_inputs,
    )
    print(
        f"[INFO] _build_render_flow_cache: built {num_frames}-frame flow cache "
        f"(N={gaussians.get_xyz.shape[0]}, dtype={dtype})."
    )


def render_set_with_clustered_deform(
    model_path,
    load2gpu_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    args,
    deform,
    use_dynamic_sep=False,
):
    """Render with clustered deform model (student models only).
    
    Output goes to standard test folder for compatibility with metrics.py.
    """
    if not isinstance(deform, ClusteredDeformModel):
        # Fall back to standard rendering if not clustered
        render_set(
            model_path, load2gpu_on_the_fly, is_6dof, name, iteration,
            views, gaussians, pipeline, background, args, deform,
            use_dynamic_sep
        )
        return
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    total_time = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()

        # LazyCamera: image starts as None; load from disk on demand
        if view.original_image is None and hasattr(view, 'load_image_to_gpu'):
            view.load_image_to_gpu('cuda')

        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        # Resolve prev_xyz for flow-mode routing.
        # When query_mode=="flow" and the displacement cache is available,
        # we look up the previous frame's deformed positions so that cluster
        # routing at render time matches the training-time behaviour.
        if deform.query_mode == "flow" and gaussians._deform_cache is not None:
            _T = gaussians._flow_total_frames
            t_idx = int(round(fid.item() * (_T - 1))) if _T > 1 else 0
            prev_xyz = gaussians.get_prev_xyz(t_idx)
        else:
            prev_xyz = None

        # ---- timing starts (deform + render only, excludes I/O) ----
        torch.cuda.synchronize()
        t_start = time.time()

        # Student models forward (clustered)
        d_xyz, d_rotation, d_scaling = deform.step(
            xyz.detach(), time_input, gaussians._cluster_labels, prev_xyz=prev_xyz
        )
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)

        torch.cuda.synchronize()
        total_time += time.time() - t_start
        # ---- timing ends ----

        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        # Free image VRAM — thousands of cameras cannot all stay resident
        if hasattr(view, 'unload_image'):
            view.unload_image()

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")


def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, args, deform, use_dynamic_sep=False):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, args, timer, use_dynamic_sep=False):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, args, deform, use_dynamic_sep=False):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, args, timer, use_dynamic_sep=False):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, args,
                              timer, use_dynamic_sep=False):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render_fastgs(view, gaussians, pipeline, background, args.mult, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        _deform_type = dataset.deform_type
        
        # Detect if using clustered deform model by scanning saved weight files.
        # This is more robust than relying on CLI args which may not be passed at render time.
        _use_clustered = False
        _use_batched = False
        if _deform_type == "4dgs":
            _deform_dir = os.path.join(dataset.model_path, "deform")
            if os.path.isdir(_deform_dir):
                for _dname in os.listdir(_deform_dir):
                    _iter_dir = os.path.join(_deform_dir, _dname)
                    if not os.path.isdir(_iter_dir):
                        continue
                    if glob.glob(os.path.join(_iter_dir, "deform_cluster_*.pth")):
                        _use_clustered = True
                        break
                    if os.path.isfile(os.path.join(_iter_dir, "batched_students.pth")):
                        _use_clustered = True
                        _use_batched = True
                        break
        
        if _deform_type == "4dgs":
            _s_res = tuple(int(x) for x in dataset.hex_spatial_res.split(","))
            _t_res = tuple(int(x) for x in dataset.hex_time_res.split(","))
            
            if _use_clustered:
                # Load ClusteredDeformModel
                n_clusters = dataset.cluster_n_clusters
                
                # Load capacity tier configs
                capacity_tier_config_path = dataset.capacity_tier_config_path
                capacity_tier_configs = None
                if os.path.exists(capacity_tier_config_path):
                    with open(capacity_tier_config_path, 'r') as f:
                        capacity_tier_configs = json.load(f)
                
                # Infer student configs from weight files
                deform_dir = os.path.join(dataset.model_path, "deform")

                if _use_batched:
                    # Batched mode: infer uniform architecture from the saved state dict shapes.
                    # No per-cluster tier files exist — all config is encoded in tensor shapes.
                    _max_iter = -1
                    if os.path.isdir(deform_dir):
                        for _dn in os.listdir(deform_dir):
                            _m = re.match(r"iteration_(\d+)", _dn)
                            if _m:
                                _max_iter = max(_max_iter, int(_m.group(1)))
                    _batched_pth = os.path.join(
                        deform_dir, f"iteration_{_max_iter}", "batched_students.pth"
                    )
                    _sd = torch.load(_batched_pth, map_location="cpu")
                    # hexplane.planes.{lvl}.2 is the XT plane: (K, feat_dim, spatial_res, time_res)
                    _num_levels = len(set(
                        k.split(".")[2]
                        for k in _sd if k.startswith("hexplane.planes.") and k.endswith(".0")
                    ))
                    _spatial_res, _time_res = [], []
                    for _lvl in range(_num_levels):
                        _pXT = _sd[f"hexplane.planes.{_lvl}.2"]  # (K, C, s_res, t_res)
                        _spatial_res.append(_pXT.shape[2])
                        _time_res.append(_pXT.shape[3])
                    _K = _sd["hexplane.planes.0.0"].shape[0]
                    _feat_dim = _sd["hexplane.planes.0.0"].shape[1]
                    _mlp_hidden = _sd["decoder.W0"].shape[1]
                    _mlp_layers = 1 + sum(1 for k in _sd if k.startswith("decoder.W_mid.") and k.count(".") == 2)
                    n_clusters = _K
                    _uniform_cfg = {
                        "spatial_resolutions": _spatial_res,
                        "time_resolutions": _time_res,
                        "feat_dim": _feat_dim,
                        "mlp_hidden_dim": _mlp_hidden,
                        "mlp_layer_num": _mlp_layers,
                        "tier": "high",
                    }
                    student_configs = [_uniform_cfg] * n_clusters
                    print(
                        f"[INFO] Batched mode: K={_K}, spatial={_spatial_res}, "
                        f"feat_dim={_feat_dim}, mlp_hidden={_mlp_hidden}, mlp_layers={_mlp_layers}"
                    )

                elif os.path.isdir(deform_dir):
                    # Sequential mode: infer per-cluster configs from tier labels in filenames
                    iter_pattern = re.compile(r"iteration_(\d+)")
                    max_iter = -1
                    for dirname in os.listdir(deform_dir):
                        match = iter_pattern.match(dirname)
                        if match:
                            iter_num = int(match.group(1))
                            if iter_num > max_iter:
                                max_iter = iter_num

                    if max_iter >= 0:
                        iter_dir = os.path.join(deform_dir, f"iteration_{max_iter}")
                        dual_pattern = re.compile(
                            r"deform_cluster_hex(?P<hex_tier>high|medium|low)_mlp(?P<mlp_tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
                        )
                        single_pattern = re.compile(
                            r"deform_cluster_(?P<tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
                        )
                        cluster_tiers = {}
                        for filename in sorted(os.listdir(iter_dir)):
                            m = dual_pattern.match(filename)
                            if m:
                                cid = int(m.group("cluster_id"))
                                cluster_tiers[cid] = {
                                    "hex_tier": m.group("hex_tier"),
                                    "mlp_tier": m.group("mlp_tier"),
                                }
                                continue
                            m = single_pattern.match(filename)
                            if m:
                                cid = int(m.group("cluster_id"))
                                cluster_tiers[cid] = m.group("tier")

                        student_configs = infer_student_configs_from_weights(
                            cluster_tiers=cluster_tiers,
                            n_clusters=n_clusters,
                            capacity_tier_configs=capacity_tier_configs
                        )
                        print(f"[INFO] Inferred student configs from weight files: {len(student_configs)} clusters")
                    else:
                        print("[WARNING] No iteration directory found, using default student configs")
                        student_configs = None
                else:
                    print("[WARNING] No deform directory found, using default student configs")
                    student_configs = None

                # Create ClusteredDeformModel with inferred configs
                deform = ClusteredDeformModel(
                    n_clusters=n_clusters,
                    is_blender=dataset.is_blender,
                    is_6dof=dataset.is_6dof,
                    student_configs=student_configs,
                    use_batched_students=_use_batched,
                    query_mode=getattr(args, 'query_mode', 'canonical'),
                    routing_mode=getattr(args, 'routing_mode', 'hard'),
                    soft_overlap_ratio=getattr(args, 'soft_overlap_ratio', 1.3),
                    soft_routing_k=getattr(args, 'soft_routing_k', 2),
                )
                deform.load_weights(dataset.model_path, iteration if iteration >= 0 else -1)
                print(f"[INFO] Loaded clustered deform model with {n_clusters} student models (batched={_use_batched})")
                
                # Set cluster labels from loaded Gaussians (saved in PLY)
                if hasattr(gaussians, '_cluster_labels') and gaussians._cluster_labels is not None:
                    deform.set_cluster_labels(gaussians._cluster_labels)
                    print(f"[INFO] Loaded cluster labels: {gaussians._cluster_labels.shape[0]} Gaussians")
                else:
                    print("[WARNING] No cluster labels found in loaded point cloud!")

                # ---- flow-mode setup (cluster_centers + displacement cache) ----
                if (
                    isinstance(deform, ClusteredDeformModel)
                    and deform.query_mode == "flow"
                    and hasattr(gaussians, '_cluster_labels')
                    and gaussians._cluster_labels is not None
                ):
                    # cluster_centers are required for flow routing; they are NOT
                    # saved in the checkpoint, so we recompute them from canonical positions.
                    # At render time there are no deform stats → canonical AABB only.
                    _aabb_pad = getattr(dataset, 'cluster_aabb_padding', 0.15)
                    deform.set_per_cluster_aabb(
                        gaussians.get_xyz.detach(),
                        cluster_labels=gaussians._cluster_labels,
                        padding=_aabb_pad,
                    )
                    print(f"[INFO] Flow mode: cluster_centers set for {deform.n_clusters} clusters.")

                    # Build the per-frame displacement cache using trained students
                    # in canonical mode, so render pass can look up prev_xyz.
                    _flow_dtype = torch.float16
                    if getattr(args, 'flow_cache_dtype', 'float16') == 'float32':
                        _flow_dtype = torch.float32
                    _n_frames = getattr(dataset, 'num_images', 300)
                    _build_render_flow_cache(
                        gaussians, deform,
                        num_frames=_n_frames,
                        dtype=_flow_dtype,
                    )
                # ---- end flow-mode setup ----
            else:
                # Load standard DeformModel_4DGS
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
                deform.load_weights(dataset.model_path)
        else:
            deform = DeformModel(dataset.is_blender, dataset.is_6dof)
            deform.load_weights(dataset.model_path)

        _use_dynamic_sep = dataset.use_dynamic_sep
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # For clustered deform model, use dual-path rendering
        if _use_clustered:
            render_func = render_set_with_clustered_deform
            print("[INFO] Using dual-path rendering for clustered deform model")
        elif mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, args, deform, use_dynamic_sep=_use_dynamic_sep)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, args, deform, use_dynamic_sep=_use_dynamic_sep)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optim = OptimizationParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)

    # Backfill defaults if missing from cfg_args
    _optim_defaults = OptimizationParams(ArgumentParser())
    for _k in ["mult"]:
        if not hasattr(args, _k) or getattr(args, _k) is None:
            setattr(args, _k, getattr(_optim_defaults, _k))

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args)
