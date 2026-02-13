"""
PyTorch wrapper for Optical Flow Rasterization via diff_flow_rasterization.

Provides:
  - FlowRasterizerHelper: convenience nn.Module around the CUDA extension
  - OpticalFlowLoss: masked L1 + optional TV loss
  - Utility functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from diff_flow_rasterization import (
    FlowRasterizationSettings,
    FlowRasterizer,
)


# ──────────────────────────────────────────────────────────────
# High-level helper that wraps FlowRasterizer for training use
# ──────────────────────────────────────────────────────────────

class FlowRasterizerHelper(nn.Module):
    """
    Convenience wrapper that:
      1. Builds FlowRasterizationSettings from camera parameters.
      2. Detaches geometry so gradients only flow to velocity3D.
      3. Calls the CUDA flow rasterizer.

    Usage in training loop::

        helper = FlowRasterizerHelper()
        flow_pred, radii, depth = helper.render_flow(
            gaussians=pc,
            velocity3D=deform_net.velocity,
            viewpoint_camera=cam,
        )
    """

    def __init__(self, bg_color: Optional[torch.Tensor] = None,
                 scale_modifier: float = 1.0,
                 mult: float = 1.0,
                 debug: bool = False):
        super().__init__()
        if bg_color is None:
            bg_color = torch.zeros(2)       # 2-channel flow background
        self.register_buffer("bg_color", bg_color.float())
        self.scale_modifier = scale_modifier
        self.mult = mult
        self.debug = debug

    def render_flow(
        self,
        gaussians,                      # GaussianModel
        velocity3D: torch.Tensor,       # [P, 3]
        viewpoint_camera,               # Camera object (MiniCam / similar)
        override_means3D: Optional[torch.Tensor] = None,
        detach_geometry: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render per-pixel 2D optical flow via alpha blending of Jacobian-projected velocities.

        Args:
            gaussians: GaussianModel containing xyz / scaling / rotation / opacity.
            velocity3D: [P, 3] world-space velocity (requires_grad=True).
            viewpoint_camera: camera with .image_height, .image_width,
                .FoVx, .FoVy, .world_view_transform, .full_proj_transform.
            override_means3D: optional override for Gaussian means.
            detach_geometry: if True, detach geometry so only velocity3D receives grads.

        Returns:
            flow:  [2, H, W]
            radii: [P]
            depth: [1, H, W]
        """
        bg = self.bg_color.to(velocity3D.device)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = FlowRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=self.scale_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            mult=self.mult,
            prefiltered=False,
            debug=self.debug,
        )

        rasterizer = FlowRasterizer(raster_settings=raster_settings)

        # ── geometry ──
        means3D = override_means3D if override_means3D is not None else gaussians.get_xyz
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        opacity = gaussians.get_opacity

        if detach_geometry:
            means3D = means3D.detach()
            scales = scales.detach()
            rotations = rotations.detach()
            opacity = opacity.detach()

        # screenspace placeholder (same pattern as RGB rasterizer)
        screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        flow, radii, depth = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            opacities=opacity,
            velocity3D=velocity3D,
            scales=scales,
            rotations=rotations,
        )

        return flow, radii, depth


class OpticalFlowLoss(nn.Module):
    """
    Loss function for optical flow supervision.
    Combines L1 loss on valid pixels with optional TV regularization.
    """

    def __init__(self, use_tv_loss: bool = False, tv_weight: float = 0.01):
        super().__init__()
        self.use_tv_loss = use_tv_loss
        self.tv_weight = tv_weight

    def forward(
        self,
        flow_pred: torch.Tensor,       # [2, H, W]
        flow_gt: torch.Tensor,         # [2, H, W]
        valid_mask: torch.Tensor,       # [1, H, W] or [H, W]
    ) -> torch.Tensor:
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(0)

        diff = torch.abs(flow_pred - flow_gt)
        l1_loss = (diff * valid_mask).sum() / (valid_mask.sum() * flow_pred.shape[0] + 1e-8)

        if self.use_tv_loss:
            tv = self._tv(flow_pred) * self.tv_weight
            return l1_loss + tv
        return l1_loss

    @staticmethod
    def _tv(flow: torch.Tensor) -> torch.Tensor:
        """Total variation on [C, H, W] tensor."""
        gx = torch.abs(flow[:, 1:, :] - flow[:, :-1, :]).mean()
        gy = torch.abs(flow[:, :, 1:] - flow[:, :, :-1]).mean()
        return gx + gy


# ── Utility ──

def compute_flow_magnitude(flow: torch.Tensor) -> torch.Tensor:
    """
    Args:
        flow: [2, H, W] or [H, W, 2]
    Returns:
        [H, W] magnitude
    """
    if flow.shape[-1] == 2:
        flow = flow.permute(2, 0, 1)
    return torch.norm(flow, dim=0)

