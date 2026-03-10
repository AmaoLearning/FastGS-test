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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def flow_dynamic_supervision_loss(
    prob_map: torch.Tensor,
    flow_gt: torch.Tensor,
    flow_mask: torch.Tensor,
    flow_thresh: float = 3.0,
) -> torch.Tensor:
    """Supervised dynamic-prob loss using optical flow magnitude as soft target.

    For each pixel, the target dynamic probability is derived from the GT flow
    magnitude:  ``target = clamp(||flow_gt|| / flow_thresh, 0, 1)``.

    A masked BCE loss is computed between the rendered prob map and this target.

    Args:
        prob_map: (1, H, W) or (3, H, W)  Alpha-blended dynamic prob image.
        flow_gt:  (2, H, W)  Ground-truth optical flow.
        flow_mask: (1, H, W) boolean validity mask.
        flow_thresh: Flow magnitude (px) that maps to target=1.  Motions below
            this threshold get a proportionally lower target.
    """
    # Soft target from flow magnitude
    flow_mag = flow_gt.norm(dim=0, keepdim=True)           # (1, H, W)
    target = (flow_mag / max(flow_thresh, 1e-6)).clamp(0, 1)  # (1, H, W)

    pred = prob_map[:1]  # take first channel, (1, H, W)
    pred = pred.clamp(1e-6, 1.0 - 1e-6)  # numerical safety for BCE

    mask = flow_mask[:1].bool() if flow_mask is not None else torch.ones_like(pred, dtype=torch.bool)
    valid = mask.sum()
    if valid == 0:
        return pred.new_zeros(())

    return F.binary_cross_entropy(pred[mask], target[mask])


def dynamic_sparsity_loss(prob: torch.Tensor) -> torch.Tensor:
    """Sparsity prior: encourage most Gaussians to be static (prob → 0).

    L_sparse = mean(p_i)
    """
    return prob.mean()


def gate_deform_consistency_loss(
    prob: torch.Tensor,
    d_xyz_raw: torch.Tensor,
) -> torch.Tensor:
    """Gate–deform consistency: if raw deformation is large, prob must be high.

    L_gate = mean( (1 - p_i) * ||stop_grad(d_xyz_raw_i)||_2 )

    The stop-gradient on d_xyz_raw ensures this loss only trains dynamic_logit
    without disturbing the deformation network.
    """
    deform_mag = d_xyz_raw.detach().norm(dim=-1, keepdim=True)  # (N, 1)
    return ((1.0 - prob) * deform_mag).mean()


def binary_entropy_polarization_loss(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean binary entropy over a batch of probabilities.

    Minimising this loss pushes each element of *prob* toward 0 or 1,
    producing a bimodal (polarised) distribution.

    Args:
        prob: Tensor of shape ``(N, 1)`` or ``(N,)`` with values in ``(0, 1)``.
        eps: Clamping margin to avoid ``log(0)``.

    Returns:
        Scalar mean binary entropy.
    """
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return F.mse_loss(network_output, gt)


def velocity_temporal_smoothness_loss(
    velocity_network, xyz: torch.Tensor, t: torch.Tensor,
    dt: float = 0.01, v_t: torch.Tensor = None,
):
    """
    计算速度场的时间平滑正则化损失。
    
    通过约束相邻时间步的速度场变化来确保时间上的平滑性，
    即 ||v(x, t+dt) - v(x, t)||^2 应该较小。
    
    Args:
        velocity_network: 速度场网络
        xyz: 高斯中心点坐标 [N, 3]
        t: 当前时间 [N, 1]
        dt: 时间采样间隔 (默认 0.01)
        v_t: 预计算的当前时刻速度 [N, 3]，如果提供则跳过一次 forward
    
    Returns:
        时间平滑正则化损失 (标量)
    """
    # 计算当前时间步的速度 — reuse if pre-computed
    if v_t is None:
        v_t = velocity_network(xyz, t)
    
    # 计算下一时间步的速度 (t + dt)
    t_next = t + dt
    v_t_next = velocity_network(xyz, t_next)
    
    # 计算速度变化的 L2 范数
    smooth_loss = ((v_t_next - v_t) ** 2).mean()
    
    return smooth_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
