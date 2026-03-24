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

import os
import time
import logging
import traceback
import torch
import math
from random import randint
from utils.loss_utils import (l1_loss, ssim, kl_divergence, l2_loss)
from gaussian_renderer import render_fastgs, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, DeformModel_4DGS
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from fused_ssim import fused_ssim as fast_ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import random
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.cluster_utils import cluster_dynamic_gaussians, render_cluster_pseudocolor


def run_clustering_at_iteration(
    gaussians,
    scene,
    deform,
    pipe,
    background,
    dataset,
    iteration: int,
    temperature: float = 1.0,
    mult: float = 1.0,
    tb_writer=None,
):
    """Run dynamic Gaussian clustering at specified iteration (decoupled from training loop).

    Args:
        gaussians: GaussianModel instance.
        scene: Scene instance.
        deform: Deformation model.
        pipe: Pipeline params.
        background: Background color tensor.
        dataset: Dataset with clustering parameters.
        iteration: Current iteration number for naming outputs.
        temperature: Temperature for sigmoid scaling.
        mult: Scaling multiplier for rendering.
        tb_writer: TensorBoard writer (optional).
    """
    if not getattr(dataset, 'use_dynamic_sep', False):
        return None

    _cluster_save = os.path.join(
        dataset.model_path, "cluster",
        f"cluster_iter{iteration}.npz")
    os.makedirs(os.path.dirname(_cluster_save), exist_ok=True)

    cluster_result = cluster_dynamic_gaussians(
        gaussians,
        dynamic_thresh=getattr(dataset, 'cluster_dynamic_thresh', 0.5),
        n_clusters=getattr(dataset, 'cluster_n_clusters', 8),
        w_xyz=getattr(dataset, 'cluster_w_xyz', 1.0),
        w_color=getattr(dataset, 'cluster_w_color', 0.5),
        w_motion=getattr(dataset, 'cluster_w_motion', 1.0),
        temperature=temperature,
        tb_writer=tb_writer,
        iteration=iteration,
        save_path=_cluster_save,
        dynamic_score_percentile=getattr(dataset, 'dynamic_score_percentile', 80.0),
    )
    gaussians._cluster_labels = cluster_result["labels"]

    # Pseudo-color visualization
    _test_cams = scene.getTestCameras()
    _debug_cam = None
    for _c in _test_cams:
        if 'cam00' in _c.image_name and float(_c.fid.item()) < 1e-4:
            _debug_cam = _c
            break
    if _debug_cam is None:
        _sorted = sorted(_test_cams, key=lambda c: c.fid.item())
        _debug_cam = _sorted[0]
    scene.ensure_cameras_loaded([_debug_cam])
    _vis_path = os.path.join(
        dataset.model_path, "cluster",
        f"cluster_vis_iter{iteration}.png")
    render_cluster_pseudocolor(
        gaussians,
        labels=cluster_result["labels"],
        viewpoint_cam=_debug_cam,
        deform=deform,
        pipe=pipe,
        bg_color=background,
        mult=mult,
        is_6dof=dataset.is_6dof,
        save_path=_vis_path,
        tb_writer=tb_writer,
        iteration=iteration,
    )
    scene.release_cameras([_debug_cam])
    print(f"[INFO] Clustering at iter {iteration} saved to {_cluster_save}")
    return cluster_result


# ── Helper function: gradual weight scheduling ──
def _get_static_suppress_weight(current_iter: int, start_iter: int, end_iter: int) -> float:
    """Get suppression weight for static Gaussians using cosine annealing.
    
    Weight starts at 1.0 (no suppression) and decreases to 0.0 (full suppression)
    using cosine annealing to avoid abrupt changes.
    
    Args:
        current_iter: Current training iteration
        start_iter: Iteration to start suppression (default: 15000)
        end_iter: Iteration to reach full suppression (default: 25000)
    
    Returns:
        Weight in range [0, 1]: 1.0 = no suppression, 0.0 = full suppression
    """
    if current_iter <= start_iter:
        return 1.0
    if current_iter >= end_iter:
        return 0.0
    
    progress = (current_iter - start_iter) / (end_iter - start_iter)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, quiet: bool = False,
             profile: bool = False, profile_start: int = 500, profile_steps: int = 50,
             logger: logging.Logger = None):
    # Fall back to a no-op logger when none is provided
    if logger is None:
        logger = logging.getLogger("train")

    safe_state(quiet) # fix random seeds
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)

    # ── Select deformation network ──
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
        logger.info("Using 4DGS HexPlane deformation network  spatial_res=%s  time_res=%s  feat_dim=%d  mlp=%dx%d  fusion=%s",
                    _s_res, _t_res, dataset.hex_feat_dim, dataset.hex_mlp_hidden, dataset.hex_mlp_layers, dataset.hex_fusion)
        print(f"[INFO] Using 4DGS HexPlane  spatial_res={_s_res}  time_res={_t_res}  feat_dim={dataset.hex_feat_dim}  mlp={dataset.hex_mlp_hidden}x{dataset.hex_mlp_layers}  fusion={dataset.hex_fusion}")
    else:
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    if dataset.clustering_iterations.strip():
        dataset.clustering_iterations = tuple(int(x.strip()) for x in dataset.clustering_iterations.split(','))
    else:
        dataset.clustering_iterations = [15000]

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, args)

    # ── Set AABB for HexPlane normalisation (4DGS only) ──
    if _deform_type == "4dgs" and hasattr(deform, "set_aabb"):
        deform.set_aabb(gaussians.get_xyz.detach(), padding=0.1)
        logger.info("HexPlane AABB set from initial point cloud")
        print("[INFO] HexPlane AABB set from initial point cloud")

    # Initialize async image prefetch pipeline (lazy mode only)
    if scene._lazy_mode:
        scene.setup_lazy_dataloader()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Wall-clock timer — CUDA event timing was removed because
    # Event.elapsed_time() and torch.cuda.synchronize() block the CPU
    # every iteration, starving the GPU of work.
    wall_start = time.perf_counter()
    total_time = 0.0

    ema_loss_for_log = 0.0
    # best_psnr = 0.0
    # best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    # ── Phase-level timer for diagnosing GPU utilization ──
    # Records wall-clock time (with CUDA sync) per phase:
    #   data / deform / render / loss / backward / optim / other
    # Prints breakdown every 100 iterations. Adds ~0.3ms overhead per
    # iteration due to cuda.synchronize() calls — disable for production.
    _phase_accum = {k: 0.0 for k in ["data", "deform", "render", "loss_fwd", "backward", "optim", "misc"]}
    _phase_count = 0
    _PHASE_REPORT_INTERVAL = 100
    _enable_phase_timer = profile  # only active when --profile is set

    # ── torch.profiler (Chrome trace) ──
    _profiler_ctx = None
    if profile:
        trace_dir = os.path.join(dataset.model_path, "profiler_traces")
        os.makedirs(trace_dir, exist_ok=True)
        logger.info("[PROFILE] Will capture %d iterations starting at iter %d", profile_steps, profile_start)
        logger.info("[PROFILE] Trace output: %s", trace_dir)
        logger.info("[PROFILE] Phase timer enabled — prints breakdown every %d iters", _PHASE_REPORT_INTERVAL)
        print(f"[PROFILE] Will capture {profile_steps} iterations starting at iter {profile_start}")
        print(f"[PROFILE] Trace output: {trace_dir}")
        print(f"[PROFILE] Phase timer enabled — prints breakdown every {_PHASE_REPORT_INTERVAL} iters")

    for iteration in range(1, opt.iterations + 1):
        # Start profiler at the designated iteration
        if profile and iteration == profile_start and _profiler_ctx is None:
            from torch.profiler import profile as _torch_profile, ProfilerActivity, schedule, tensorboard_trace_handler
            _profiler_ctx = _torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=3, active=profile_steps, repeat=1),
                on_trace_ready=tensorboard_trace_handler(trace_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            _profiler_ctx.__enter__()
            logger.info("[PROFILE] Profiler started at iteration %d", iteration)
            print(f"[PROFILE] Profiler started at iteration {iteration}")
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_fastgs(custom_cam, gaussians, pipe, background, opt.mult, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # ── Phase timer: iteration start ──
        if _enable_phase_timer:
            torch.cuda.synchronize()

        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if _enable_phase_timer:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        viewpoint_cam, total_frame = scene.next_train_camera()
        time_interval = 1 / total_frame
        fid = viewpoint_cam.fid

        if _enable_phase_timer:
            torch.cuda.synchronize()
            _phase_accum["data"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            _d_xyz_raw = 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)

            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

            # ── Accumulate deformation for dynamic score computation ──
            # Record deformation from iteration 10000-15000 for dynamic score at iteration 15000
            _d_xyz_raw = d_xyz  # (N, 3), unmodified output from deform network
            
            # ── Deformation distribution histogram (optional) ──
            if (dataset.log_deform_hist
                    and tb_writer
                    and iteration % 3000 == 0):
                with torch.no_grad():
                    _hist_mag = d_xyz.detach().norm(dim=-1)  # (N,)
                    tb_writer.add_histogram('deform/magnitude', _hist_mag, iteration)
                    tb_writer.add_scalar('deform/mag_max', _hist_mag.max().item(), iteration)
                    tb_writer.add_scalar('deform/mag_mean', _hist_mag.mean().item(), iteration)
                    tb_writer.add_scalar('deform/mag_median', _hist_mag.median().item(), iteration)
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                        ax.hist(_hist_mag.cpu().numpy(), bins=100, color='steelblue', edgecolor='none')
                        ax.set_xlabel('Deformation magnitude')
                        ax.set_ylabel('Count')
                        ax.set_title(f'Iter {iteration}  N={_hist_mag.shape[0]}  '
                                     f'max={_hist_mag.max().item():.4f}  mean={_hist_mag.mean().item():.4f}')
                        fig.tight_layout()
                        tb_writer.add_figure('deform/magnitude_hist', fig, iteration)
                        plt.close(fig)
                    except ImportError:
                        pass  # matplotlib not available, skip figure

            # Accumulate RAW deformation for dynamic score computation.
            # Using gated d_xyz would undercount motion for prob≈0.5 Gaussians.
            if torch.is_tensor(d_xyz) and dataset.use_dynamic_sep:
                gaussians.add_deform_stats(_d_xyz_raw)
            
            # ── Start deformation tracking from iteration 10000 ──
            if dataset.use_dynamic_sep and iteration == 10000:
                gaussians.start_deform_tracking()
                print(f"[INFO] Starting deformation tracking at iteration {iteration}")
            
            # ── Dynamic-static separation ablation test: gradually suppress static Gaussians ──
            # Clustering runs at the END of iteration 15000, so suppression starts from iteration 15001
            # Use gradual cosine annealing from 15000 to 25000 to avoid abrupt changes
            if getattr(dataset, 'use_dynamic_ablation', False) and iteration > dataset.dynamic_ablation_start_iter:
                with torch.no_grad():
                    # Get dynamic mask from clustering results
                    dynamic_mask = gaussians.get_dynamic_mask_from_cluster()  # (N,) bool
                    
                    # Check if clustering has been performed and mask dimension matches
                    if dynamic_mask.sum() == 0 or dynamic_mask.shape[0] != gaussians.get_xyz.shape[0]:
                        # No clustering performed yet or dimension mismatch due to densify/prune
                        # Only warn on the first iteration after clustering (15001)
                        if iteration == dataset.dynamic_ablation_start_iter + 1:
                            print(f"[WARNING] use_dynamic_ablation=True but no cluster labels found. "
                                  f"Please ensure clustering_iterations includes {dataset.dynamic_ablation_start_iter}")
                        # Skip masking if mask is not available or dimension mismatch
                    else:
                        # Get suppression weight (1.0 at 15000 → 0.0 at 25000)
                        _use_gradual = getattr(dataset, 'ablation_use_gradual', True)
                        _end_iter = getattr(dataset, 'dynamic_ablation_end_iter', 25000)
                        
                        if _use_gradual:
                            _suppress_weight = _get_static_suppress_weight(
                                iteration, 
                                dataset.dynamic_ablation_start_iter, 
                                _end_iter
                            )
                        else:
                            _suppress_weight = 0.0  # Immediate suppression
                        
                        # Get static mask (inverse of dynamic mask)
                        static_mask = ~dynamic_mask  # (N,) bool
                        
                        # Apply gradual suppression to static Gaussians only
                        # Log on the first iteration after clustering (15001)
                        if iteration == dataset.dynamic_ablation_start_iter + 1:
                            _n_dynamic = dynamic_mask.sum().item()
                            _n_static = static_mask.sum().item()
                            _n_total = gaussians.get_xyz.shape[0]
                            _pct_dynamic = _n_dynamic / _n_total * 100
                            _msg = (f"[ABLATION] Starting gradual static suppression at iter {iteration}: "
                                    f"{_n_dynamic} dynamic / {_n_static} static / {_n_total} total, "
                                    f"weight={_suppress_weight:.4f}")
                            print(_msg)
                            logger.info(_msg)
                            # Log deformation magnitude for debugging
                            if torch.is_tensor(d_xyz):
                                _mag_before = d_xyz.norm(dim=-1).mean().item()
                                _mag_static = d_xyz[static_mask].norm(dim=-1).mean().item() if _n_static > 0 else 0.0
                                _mag_dynamic = d_xyz[dynamic_mask].norm(dim=-1).mean().item() if _n_dynamic > 0 else 0.0
                                print(f"[ABLATION DEBUG] Deformation magnitude (before suppression):")
                                print(f"  - All Gaussians: {_mag_before:.6f}")
                                print(f"  - Static only: {_mag_static:.6f}")
                                print(f"  - Dynamic only: {_mag_dynamic:.6f}")
                                print(f"[ABLATION DEBUG] Suppression weight: {_suppress_weight:.4f} "
                                      f"(1.0=no suppression, 0.0=full suppression)")
                        
                        # Apply suppression: deformation = deformation * weight for static Gaussians
                        # This is equivalent to: d_xyz = d_xyz * (1 - (1-weight) * static_mask)
                        _static_suppress_factor = 1.0 - (1.0 - _suppress_weight) * static_mask.to(dtype=d_xyz.dtype)
                        d_xyz = d_xyz * _static_suppress_factor.unsqueeze(-1)
                        
                        # Also suppress rotation and scaling
                        if torch.is_tensor(d_rotation):
                            d_rotation = d_rotation * _static_suppress_factor.unsqueeze(-1)
                        if torch.is_tensor(d_scaling):
                            d_scaling = d_scaling * _static_suppress_factor.unsqueeze(-1)

        if _enable_phase_timer:
            torch.cuda.synchronize()
            _phase_accum["deform"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        # Render
        render_pkg_re = render_fastgs(viewpoint_cam, gaussians, pipe, background, opt.mult, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        if _enable_phase_timer:
            torch.cuda.synchronize()
            _phase_accum["render"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        # Loss
        gt_image = viewpoint_cam.original_image
        if not gt_image.is_cuda:
            gt_image = gt_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        # ── Inline TensorBoard logging (training_report is commented out) ──
        if tb_writer and iteration % 100 == 0:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/ssim_loss', ssim_loss.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            tb_writer.add_scalar('train_stats/num_gaussians', gaussians._xyz.shape[0], iteration)

        # Log final total loss (after all auxiliary losses are added)
        if tb_writer and iteration % 100 == 0:
            tb_writer.add_scalar('train_loss_patches/total_loss_final', loss.item(), iteration)

        # ── HexPlane regularisation (4DGS only) ──
        if _deform_type == "4dgs" and iteration >= opt.warm_up:
            _tv_loss = deform.get_tv_loss()
            _l1_loss = deform.get_l1_loss()
            loss = loss + 1e-3 * _tv_loss + 1e-4 * _l1_loss
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train_loss_patches/hexplane_tv', _tv_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/hexplane_l1', _l1_loss.item(), iteration)

        # ── Dynamic score-based separation (replaces dynamic probability losses) ──
        # Dynamic score is computed at iteration 15000 using displacement statistics
        # from iterations 10000-15000. No online loss supervision needed.

        if _enable_phase_timer:
            torch.cuda.synchronize()
            _phase_accum["loss_fwd"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        loss.backward()

        if _enable_phase_timer:
            torch.cuda.synchronize()
            _phase_accum["backward"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        # Release VRAM.
        # Flow: unload_flow() frees per-camera flow tensors (no-op if none loaded).
        # Image: release_camera_image() drops the reference; in lazy mode the
        #   GPU buffer persists on Scene for zero-alloc reuse.
        viewpoint_cam.unload_flow()
        scene.release_camera_image(viewpoint_cam)

        with torch.no_grad():
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # Progress bar — use detached tensor for to avoid
            # implicit cudaDeviceSynchronize() on every .item() call.
            # Only call .item() once every 100 iterations for display.
            _loss_val = loss.detach()
            ema_loss_for_log = 0.4 * _loss_val + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                if iteration % 100 == 0:
                    _display_loss = ema_loss_for_log.item() if torch.is_tensor(ema_loss_for_log) else ema_loss_for_log
                else:
                    _display_loss = _display_loss if '_display_loss' in dir() else 0.0
                progress_bar.set_postfix({"Loss": f"{_display_loss:.{7}f}", "Gaussian Number": f"{gaussians._xyz.shape[0]:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            # cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time,
            #                            testing_iterations, scene, render_fastgs, (pipe, background, opt.mult), deform,
            #                            dataset.load2gpu_on_the_fly, dataset.is_6dof)
            # if iteration in testing_iterations:
            #     if cur_psnr.item() > best_psnr:
            #         best_psnr = cur_psnr.item()
            #         best_iteration = iteration

            if iteration in saving_iterations:
                logger.info("[ITER %d] Saving Gaussians", iteration)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    scene.ensure_cameras_loaded(camlist)
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof, DENSIFY=True)
                    scene.release_cameras(camlist)

                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            else:
                if iteration % opt.densification_interval == 0:
                    gaussians.zero_accums()

            if iteration % opt.final_prune_interval == 0 and iteration > opt.final_prune_from_iter and iteration < opt.final_prune_until_iter:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                scene.ensure_cameras_loaded(camlist)
                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                scene.release_cameras(camlist)

                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)

            if iteration < opt.iterations:
                deform.optimizer.step()
                deform.optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer_step(iteration)

            if _enable_phase_timer:
                torch.cuda.synchronize()
                _phase_accum["optim"] += time.perf_counter() - _t0
                _phase_count += 1
                if _phase_count % _PHASE_REPORT_INTERVAL == 0:
                    _total = sum(_phase_accum.values()) or 1e-9
                    print(f"\n[PHASE TIMER] iter {iteration}, last {_PHASE_REPORT_INTERVAL} iters avg (ms):")
                    for k in ["data", "deform", "render", "loss_fwd", "backward", "optim"]:
                        _v = _phase_accum[k] / _phase_count * 1000
                        _pct = _phase_accum[k] / _total * 100
                        print(f"  {k:>10s}: {_v:7.2f} ms  ({_pct:5.1f}%)")
                    _iter_avg = _total / _phase_count * 1000
                    print(f"  {'TOTAL':>10s}: {_iter_avg:7.2f} ms/iter  ({1000/_iter_avg:.1f} iter/s)")
                    # Reset accumulators
                    _phase_accum = {k: 0.0 for k in _phase_accum}
                    _phase_count = 0

            # Profiler step (must be called every iteration while active)
            if _profiler_ctx is not None:
                _profiler_ctx.step()
                # Stop profiler after enough steps
                if iteration >= profile_start + profile_steps + 5:  # wait+warmup+active
                    _profiler_ctx.__exit__(None, None, None)
                    _profiler_ctx = None
                    logger.info("[PROFILE] Profiler stopped at iteration %d. Trace saved to %s", iteration, trace_dir)
                    print(f"[PROFILE] Profiler stopped at iteration {iteration}. Trace saved to {trace_dir}")

            # ── Run clustering at configurable iterations (decoupled from training loop) ──
            if dataset.use_dynamic_sep and iteration in dataset.clustering_iterations:
                run_clustering_at_iteration(
                    gaussians=gaussians,
                    scene=scene,
                    deform=deform,
                    pipe=pipe,
                    background=background,
                    dataset=dataset,
                    iteration=iteration,
                    temperature=1.0,  # temperature for dynamic score thresholding
                    mult=opt.mult,
                    tb_writer=tb_writer,
                )

    # Final sync — only once at the very end
    torch.cuda.synchronize()
    total_time = time.perf_counter() - wall_start
    # print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
    logger.info("Gaussian number: %d", gaussians._xyz.shape[0])
    logger.info("Dash time: %.2fs", total_time)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Dash time: {total_time:.2f}s")

    # Flush & close TensorBoard writer so all events are written to disk
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


def prepare_output_and_logger(dataset, opt, pipe):
    model_path = dataset.model_path if dataset is not None else None
    if not model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        model_path = os.path.join("./output/", unique_str[0:10])
        if dataset is not None:
            dataset.model_path = model_path

    # Set up output folder
    print("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)

    # Persist merged cfg (CLI + grouped params)
    if args is not None:
        combined_cfg = Namespace(**vars(args))
        if dataset is not None:
            combined_cfg.__dict__.update(vars(dataset))
        if opt is not None:
            combined_cfg.__dict__.update(vars(opt))
        if pipe is not None:
            combined_cfg.__dict__.update(vars(pipe))
        with open(os.path.join(model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(combined_cfg))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


def setup_logging(model_path: str) -> logging.Logger:
    """Initialise file-based logging under *model_path*/training.log.

    Returns a :class:`logging.Logger` named ``"train"``.
    The root logger receives a :class:`logging.FileHandler` so that any logger
    in the process tree also writes to the same file.
    """
    log_dir = model_path or "./output"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.DEBUG)

    logger = logging.getLogger("train")
    logger.info("=" * 60)
    print(f"[INFO] Log file: {log_path}")
    return logger


def run_training(args, lp: ModelParams, op: OptimizationParams, pp: PipelineParams) -> None:
    """Top-level training dispatcher: logging setup → training() → error handling."""
    logger = setup_logging(args.model_path)
    logger.info("Training started — args: %s", vars(args))

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    try:
        training(
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations,
            profile=getattr(args, 'profile', False),
            profile_start=getattr(args, 'profile_start', 500),
            profile_steps=getattr(args, 'profile_steps', 50),
            logger=logger,
        )
    except Exception:
        tb_str = traceback.format_exc()
        logger.error("Training failed with exception:\n%s", tb_str)
        _log_path = os.path.join(args.model_path or "./output", "training.log")
        print(f"\n[ERROR] Training failed. See log: {_log_path}")
        sys.exit(1)

    logger.info("Training complete.")
    print("\nTraining complete.")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000,40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000,40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler for 50 iterations (outputs Chrome trace)")
    parser.add_argument("--profile_start", type=int, default=500, help="Iteration to start profiling")
    parser.add_argument("--profile_steps", type=int, default=50, help="Number of iterations to profile")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    run_training(args, lp, op, pp)
