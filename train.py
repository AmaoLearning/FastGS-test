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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
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
from utils.flow_rasterizer import FlowRasterizerHelper, OpticalFlowLoss
from utils.optic_flow_utils import load_precomputed_flow


def training(dataset, opt, pipe, testing_iterations, saving_iterations, quiet: bool = False,
             profile: bool = False, profile_start: int = 500, profile_steps: int = 50):
    safe_state(quiet) # fix random seeds
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)

    # ── Select deformation network ──
    _deform_type = getattr(dataset, "deform_type", "mlp")
    if _deform_type == "4dgs":
        deform = DeformModel_4DGS(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof)
        print("[INFO] Using 4DGS HexPlane deformation network")
    else:
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    # torch.compile: fuse small Linear+ReLU kernels in the deform MLP to
    # reduce per-layer kernel launch overhead (~8 layers × 5μs ≈ 40μs/iter).
    # Uses 'default' mode (inductor, no CUDA graphs) to tolerate dynamic
    # shapes during densification.  Guards against older PyTorch (<2.0).
    try:
        deform.deform = torch.compile(deform.deform, mode="default", dynamic=True)
        print("[INFO] Deform network compiled with torch.compile (inductor)")
    except Exception as _e:
        print(f"[INFO] torch.compile unavailable: {_e}, using eager mode")

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, args)

    # ── Set AABB for HexPlane normalisation (4DGS only) ──
    if _deform_type == "4dgs" and hasattr(deform, "set_aabb"):
        deform.set_aabb(gaussians.get_xyz.detach(), padding=0.1)
        print("[INFO] HexPlane AABB set from initial point cloud")

    # Initialize async image prefetch pipeline (lazy mode only)
    if scene._lazy_mode:
        scene.setup_lazy_dataloader()

    # ── Optical Flow Loss Setup (deform finite-diff → diff-flow-rasterization) ──
    # flow infrastructure is needed when EITHER flow loss or flow mask is enabled
    _need_flow = dataset.use_flow_loss or dataset.use_flow_mask
    flow_helper = None
    flow_loss_fn = None
    if _need_flow:
        # 检查训练集是否有光流文件路径（延迟加载，此时不读取数组）
        any_has_flow = any(c.has_flow for c in scene.getTrainCameras())
        if any_has_flow:
            flow_helper = FlowRasterizerHelper(
                bg_color=torch.zeros(2, device="cuda"),
                scale_modifier=1.0,
                mult=opt.mult,
                debug=False,
            ).cuda()
            flow_loss_fn = OpticalFlowLoss(
                use_tv_loss=dataset.use_flow_tv_loss,
                tv_weight=opt.flow_tv_weight,
            )
            n_with_flow = sum(1 for c in scene.getTrainCameras() if c.has_flow)
            print(f"[INFO] Optical flow available for {n_with_flow}/{len(scene.getTrainCameras())} training cameras (lazy loading)")
        else:
            print(f"[WARNING] use_flow_loss/use_flow_mask=True but no flow files found, disabling.")
            dataset.use_flow_loss = False
            dataset.use_flow_mask = False
            _need_flow = False

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
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)

            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

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

        # ── Optical Flow Loss (deform finite-diff → projected flow → GT comparison) ──
        # Runs when EITHER flow loss or flow mask is enabled (mask needs per-gaussian error)
        flow_loss = None
        per_gaussian_flow_error = None
        if (_need_flow
            and flow_helper is not None
            and iteration >= opt.flow_loss_from_iter
            and iteration % opt.flow_loss_interval == 0
            and iteration >= opt.warm_up
            and torch.is_tensor(d_xyz)
            and viewpoint_cam.has_flow):
            # On-demand flow loading (consistency + magnitude mask)
            viewpoint_cam.load_flow(
                device='cuda',
                flow_magnitude_thresh=opt.flow_magnitude_thresh,
            )
            flow_gt = viewpoint_cam.flow_fwd  # [2, H, W]

            # Deform at t+dt → finite-difference displacement (NOT detached → gradient flows to deform)
            # 光流 GT = 帧间像素位移，对应 3D 量是位置形变差 d_xyz(t+dt) - d_xyz(t)，
            # 经 Jacobian 投影即得投影光流，无需除以 time_interval（那会变成速度，量纲不匹配）
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz_next, _, _ = deform.step(
                gaussians.get_xyz.detach(),
                time_input + ast_noise + time_interval
            )
            displacement3D = d_xyz_next - d_xyz  # [N, 3], 3D 位移, gradient → deform

            # Render projected optical flow
            deformed_means3D = gaussians.get_xyz + d_xyz
            flow_pred, _, _ = flow_helper.render_flow(
                gaussians=gaussians,
                velocity3D=displacement3D,
                viewpoint_camera=viewpoint_cam,
                override_means3D=deformed_means3D,
                detach_geometry=opt.detach_flow_geometry,
            )

            # Flow mask (forward-backward consistency + magnitude threshold)
            flow_mask = viewpoint_cam.flow_mask  # [1, H, W] bool or None
            if flow_mask is None:
                flow_mask = torch.ones(1, flow_gt.shape[1], flow_gt.shape[2],
                                       dtype=torch.bool, device=flow_gt.device)
            flow_loss = flow_loss_fn(flow_pred, flow_gt.float(), flow_mask.float())

            # Per-gaussian flow error for densification mask:
            # Gradient magnitude of flow_loss w.r.t. displacement3D indicates how much
            # each gaussian's deformation needs to change to reduce flow error.
            # Low magnitude → good fit → eligible for densification.
            if flow_loss.requires_grad and displacement3D.requires_grad:
                [flow_grad] = torch.autograd.grad(
                    flow_loss, displacement3D, retain_graph=True
                )
                per_gaussian_flow_error = flow_grad.detach().norm(dim=-1, keepdim=True)

            # Only add flow loss to training objective when use_flow_loss is enabled
            if dataset.use_flow_loss:
                loss = loss + opt.lambda_flow * flow_loss

            if iteration % 1000 == 0:
                print(f"[Iter {iteration}] flow loss = {flow_loss.item():.6f}")
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train_loss_patches/flow_loss', flow_loss.item(), iteration)
        
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
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # 累积 flow loss 统计 (用于 densification 掩码)
                if dataset.use_flow_mask and per_gaussian_flow_error is not None:
                    gaussians.add_flow_loss_stats(per_gaussian_flow_error, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    scene.ensure_cameras_loaded(camlist)
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof, DENSIFY=True)
                    scene.release_cameras(camlist)
                    
                    # 生成 flow_loss 掩码并传入 densification
                    flow_mask = None
                    if dataset.use_flow_mask and iteration >= opt.flow_loss_from_iter:
                        flow_mask = gaussians.get_flow_loss_mask(
                            opt.flow_loss_thresh, 
                            adaptive_percentile=opt.flow_loss_percentile
                        )
                    
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score,
                                                flow_mask = flow_mask)

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
                    print(f"[PROFILE] Profiler stopped at iteration {iteration}. Trace saved to {trace_dir}")

    # Final sync — only once at the very end
    torch.cuda.synchronize()
    total_time = time.perf_counter() - wall_start
    # print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Dash time: {total_time:.2f}s")


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

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
            profile=getattr(args, 'profile', False),
            profile_start=getattr(args, 'profile_start', 500),
            profile_steps=getattr(args, 'profile_steps', 50))

    # All done
    print("\nTraining complete.")
