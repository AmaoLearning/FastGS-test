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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render_fastgs, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
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
from utils.motion_utils import VelocityNetwork


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    # velocity
    if dataset.use_velocity:
        velocity = VelocityNetwork(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof).cuda()
        velocity.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # best_psnr = 0.0
    # best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
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

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)
        if dataset.use_velocity:
            velocity.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        velocity_loss = None  # 仅在需要时计算
        per_gaussian_velocity_loss = None  # 每个高斯的 velocity loss
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            
            # 如果启用动态掩码，只对动态高斯计算 deform
            if opt.use_dynamic_mask and iteration % opt.velocity_interval != 0:
                dynamic_mask = gaussians.get_dynamic_mask(
                    opt.dynamic_thresh, 
                    opt.grad_abs_thresh, 
                    iteration,
                    adaptive_percentile=opt.dynamic_thresh_percentile
                )
                # 初始化为零
                d_xyz = torch.zeros((N, 3), device="cuda")
                d_rotation = torch.zeros((N, 4), device="cuda")
                d_scaling = torch.zeros((N, 3), device="cuda")
                # 只对动态高斯计算 deform
                if dynamic_mask.sum() > 0:
                    d_xyz_masked, d_rotation_masked, d_scaling_masked = deform.step(
                        gaussians.get_xyz[dynamic_mask].detach(), 
                        time_input[dynamic_mask] + (ast_noise[dynamic_mask] if torch.is_tensor(ast_noise) else ast_noise)
                    )
                    d_xyz[dynamic_mask] = d_xyz_masked
                    d_rotation[dynamic_mask] = d_rotation_masked
                    d_scaling[dynamic_mask] = d_scaling_masked
                
                if iteration % 1000 == 0:
                    print(f"[Iter {iteration}] Dynamic mask: {dynamic_mask.sum().item()}/{N} ({100*dynamic_mask.sum().item()/N:.2f}%)")
            else:
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
            
            if dataset.use_velocity and iteration % opt.velocity_interval == 0:
                _d_xyz, _, _ = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise + time_interval)
                current_v = velocity.forward(gaussians.get_xyz.detach(), time_input + ast_noise)
                
                # 更新动态指标 (Leaky Max)
                with torch.no_grad():
                    gaussians.update_dynamic_metrics(current_v.detach(), decay=opt.dynamic_decay)
                
                # 计算每个高斯的 velocity loss (用于累积统计)
                velocity_diff = _d_xyz - d_xyz
                velocity_pred = current_v * time_interval
                per_gaussian_velocity_loss = ((velocity_diff - velocity_pred) ** 2).mean(dim=-1, keepdim=True)
                
                # 总体 velocity loss (用于反向传播)
                velocity_loss = per_gaussian_velocity_loss.mean()

                if iteration % 1000 == 0:
                    print(f"[Iter {iteration}] velocity loss = {velocity_loss.item():.6f}")
                    print(f"[Iter {iteration}] velocity mean = {current_v.mean(dim=0).detach().cpu().numpy()} at time {(time_input + ast_noise)[0].item():.4f}")

        # Render
        render_pkg_re = render_fastgs(viewpoint_cam, gaussians, pipe, background, opt.mult, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        
        # 只有在 use_velocity 开启且不在 warm_up 期间时才加入 velocity_loss
        if dataset.use_velocity and iteration >= opt.warm_up and velocity_loss is not None:
            loss = loss + opt.lambda_velocity * velocity_loss
        
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Gaussian Number": f"{gaussians._xyz.shape[0]:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            iter_time = iter_start.elapsed_time(iter_end)
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
            optim_start.record()
            if iteration < opt.densify_until_iter:
                # 累积 velocity loss 统计 (用于 densification 掩码)
                if dataset.use_velocity and per_gaussian_velocity_loss is not None:
                    gaussians.add_velocity_loss_stats(per_gaussian_velocity_loss.detach(), visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # The multiview consistent densification of fastgs
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof, DENSIFY=True)
                    
                    # 生成 velocity_loss 掩码并传入 densification
                    velocity_mask = None
                    if dataset.use_velocity:
                        velocity_mask = gaussians.get_velocity_loss_mask(
                            opt.velocity_loss_thresh, 
                            adaptive_percentile=opt.velocity_loss_percentile
                        )
                    
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score,
                                                velocity_mask = velocity_mask)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            else:
                if iteration % opt.densification_interval == 0:
                    gaussians.zero_accums()

            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof)                    
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
            
            if iteration < opt.iterations:
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                if dataset.use_velocity:
                    velocity.optimizer.step()
                    velocity.optimizer.zero_grad()
                gaussians.optimizer_step(iteration)

            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Dash time: {total_time}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
