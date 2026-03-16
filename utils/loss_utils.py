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
    invalid_weight: float = 0.2,
    target_gamma: float = 1.5,
) -> torch.Tensor:
    """Flow-supervised dynamic-prob loss with soft consistency weighting.

    Best-practice choices here:
      1) Use flow magnitude as *soft* dynamic target.
      2) Keep consistency info as confidence (weight), not hard masking.
         This avoids supervision holes while still down-weighting unreliable areas.

    Args:
        prob_map: (1, H, W) or (3, H, W), rendered dynamic probability map.
        flow_gt: (2, H, W), GT optical flow.
        flow_mask: (1, H, W) bool/float consistency mask, optional.
        flow_thresh: Motion scale (px) used in target normalization.
        invalid_weight: Minimum weight for low-confidence pixels in [0, 1].
        target_gamma: Gamma > 1 suppresses tiny motions; < 1 amplifies them.
    """
    pred = prob_map[:1].float().clamp(1e-6, 1.0 - 1e-6)  # (1, H, W)

    # Soft target from motion magnitude; gamma suppresses camera/noise micro-motion.
    flow_mag = flow_gt.float().norm(dim=0, keepdim=True)  # (1, H, W)
    target = (flow_mag / max(flow_thresh, 1e-6)).clamp(0, 1)
    target = target.pow(max(target_gamma, 1e-6))

    # Confidence weight map: avoid hard holes by assigning a non-zero floor weight.
    if flow_mask is None:
        conf = torch.ones_like(target)
    else:
        conf = flow_mask[:1].float().clamp(0.0, 1.0)
    base_w = float(max(0.0, min(1.0, invalid_weight)))
    weight = base_w + (1.0 - base_w) * conf

    bce = F.binary_cross_entropy(pred, target, reduction="none")
    wsum = weight.sum().clamp_min(1e-6)
    return (bce * weight).sum() / wsum


def flow_classify_bce_loss(
    classifier_logits: torch.Tensor,
    flow_gt: torch.Tensor,
    binarize_percentile: float = 50.0,
    dual_thresh_low: float = 30.0,
    dual_thresh_high: float = 70.0,
) -> torch.Tensor:
    """BCE loss between 2D classifier logits and percentile-binarised optical flow.

    Pipeline (SA4D-inspired):
      1) ``flow_gt → norm → flow_mag``
      2) Dual percentile threshold → ``reliability_mask``
      3) Percentile binarisation → ``binary_target`` (hard 0/1 label)
      4) ``BCE_with_logits(logits, target)`` weighted by reliability mask

    All three percentile thresholds are resolved from a **single**
    ``torch.sort`` call — O(n log n) once, then three O(1) index lookups.

    Args:
        classifier_logits: ``(1, H, W)`` raw logits from :class:`DynamicClassifier2D`.
        flow_gt: ``(2, H, W)`` optical flow (forward).
        binarize_percentile: Percentile (0–100) of flow magnitude used as the
            dynamic/static binarisation cut-off.
        dual_thresh_low: Percentile (0–100) below which pixels are *reliably static*.
        dual_thresh_high: Percentile (0–100) above which pixels are *reliably dynamic*.
    """
    # 1) Flow magnitude
    flow_mag = flow_gt.float().norm(dim=0, keepdim=True)  # (1, H, W)

    # 2-3) Single sort → all three percentile values in one pass
    _sorted = flow_mag.flatten().sort().values
    _n = _sorted.numel()
    _idx_lo = max(0, min(int(dual_thresh_low / 100.0 * _n), _n - 1))
    _idx_mid = max(0, min(int(binarize_percentile / 100.0 * _n), _n - 1))
    _idx_hi = max(0, min(int(dual_thresh_high / 100.0 * _n), _n - 1))
    val_lo = _sorted[_idx_lo]
    pct_val = _sorted[_idx_mid]
    val_hi = _sorted[_idx_hi]

    # Dual-percentile reliability mask
    reliable_static = flow_mag < val_lo       # definitely static
    reliable_dynamic = flow_mag > val_hi      # definitely dynamic
    reliable_mask = (reliable_static | reliable_dynamic).float()

    # Percentile binarisation → hard target
    binary_target = (flow_mag > pct_val).float()
    # Enforce consistency: reliable-region target must agree with threshold class
    binary_target = torch.where(reliable_dynamic, torch.ones_like(binary_target), binary_target)
    binary_target = torch.where(reliable_static, torch.zeros_like(binary_target), binary_target)

    # 4) Weighted BCE (logits → numerically stable)
    logits = classifier_logits[:1].float()
    bce = F.binary_cross_entropy_with_logits(logits, binary_target, reduction="none")
    wsum = reliable_mask.sum().clamp_min(1.0)
    return (bce * reliable_mask).sum() / wsum


def compute_knn_indices(xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Compute k-nearest-neighbor indices via scipy cKDTree.

    Runs on CPU (O(n log n) KD-tree build + O(n k log n) query).
    Returns a CUDA ``LongTensor`` of shape ``(N, k)``.
    """
    from scipy.spatial import cKDTree
    pts = xyz.detach().cpu().numpy()
    tree = cKDTree(pts)
    # k+1 because the closest point is the query itself
    _, indices = tree.query(pts, k=k + 1, workers=-1)
    indices = indices[:, 1:]  # drop self
    return torch.from_numpy(indices).long().to(xyz.device)


def spatial_kl_regularization_loss(
    dynamic_logit: torch.Tensor,
    knn_indices: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """3D spatial KL regularization (SA4D-inspired).

    For each Gaussian *i* with probability ``p_i = sigmoid(logit_i / T)``,
    compute the mean probability ``q_i`` of its *k* nearest neighbors,
    then average the Bernoulli KL divergence:

    .. math::

        \\mathcal{L} = \\frac{1}{N} \\sum_i
            \\mathrm{KL}\\bigl(\\mathrm{Bern}(p_i) \\|\n            \\mathrm{Bern}(q_i)\\bigr)

    This encourages spatial consistency: nearby Gaussians should share
    similar dynamic/static identity, suppressing isolated outliers.

    Args:
        dynamic_logit: ``(N, 1)`` raw logits.
        knn_indices: ``(N, k)`` LongTensor of neighbor indices.
        temperature: Sigmoid temperature (same as the gating temperature).
        eps: Clamping margin to avoid ``log(0)``.
    """
    p = torch.sigmoid(dynamic_logit.squeeze(-1) / max(temperature, 1e-8))  # (N,)
    p = p.clamp(eps, 1.0 - eps)

    # Gather neighbor probabilities → mean
    neighbor_p = p[knn_indices]              # (N, k)
    q = neighbor_p.mean(dim=1).clamp(eps, 1.0 - eps)  # (N,)

    # Bernoulli KL(p || q)
    kl = p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())
    return kl.mean()


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
