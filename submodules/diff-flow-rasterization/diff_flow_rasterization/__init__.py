"""
diff_flow_rasterization/__init__.py
Mirrors diff_gaussian_rasterization_fastgs/__init__.py
Adapted for 2-channel flow rasterization with velocity3D input.
No SH / dc / campos / metric_map / adamUpdate.
"""

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_flow(
    means3D,
    means2D,
    velocity3D,
    flow_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeFlow.apply(
        means3D,
        means2D,
        velocity3D,
        flow_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeFlow(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        velocity3D,
        flow_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        args = (
            raster_settings.bg,
            means3D,
            velocity3D,
            flow_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.mult,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, num_buckets, flow, radii, depth, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_flow(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_flow_fw.dump")
                print("\nAn error occurred in flow forward. Saved snapshot_flow_fw.dump.\n")
                raise ex
        else:
            num_rendered, num_buckets, flow, radii, depth, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_flow(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(
            velocity3D, flow_precomp, means3D, scales, rotations,
            cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer
        )
        return flow, radii, depth

    @staticmethod
    def backward(ctx, grad_out_flow, _, grad_out_depth):
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        (velocity3D, flow_precomp, means3D, scales, rotations,
         cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer) = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            velocity3D,
            flow_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_flow,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            num_buckets,
            sampleBuffer,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (grad_means2D, grad_flow, grad_opacities,
                 grad_means3D, grad_velocity3D,
                 grad_scales, grad_rotations) = _C.rasterize_flow_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_flow_bw.dump")
                print("\nAn error occurred in flow backward. Saved snapshot_flow_bw.dump.\n")
                raise ex
        else:
            (grad_means2D, grad_flow, grad_opacities,
             grad_means3D, grad_velocity3D,
             grad_scales, grad_rotations) = _C.rasterize_flow_backward(*args)

        grads = (
            grad_means3D,       # means3D
            grad_means2D,       # means2D
            grad_velocity3D,    # velocity3D
            grad_flow,          # flow_precomp
            grad_opacities,     # opacities
            grad_scales,        # scales
            grad_rotations,     # rotations
            None,               # cov3Ds_precomp  (TODO: if needed)
            None,               # raster_settings
        )

        return grads


class FlowRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    mult: float
    prefiltered: bool
    debug: bool


class FlowRasterizer(nn.Module):
    def __init__(self, raster_settings: FlowRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )
        return visible

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        velocity3D: torch.Tensor = None,
        flow_precomp: torch.Tensor = None,
        scales: torch.Tensor = None,
        rotations: torch.Tensor = None,
        cov3D_precomp: torch.Tensor = None,
    ):
        raster_settings = self.raster_settings

        if (velocity3D is None and flow_precomp is None) or \
           (velocity3D is not None and flow_precomp is not None):
            raise Exception(
                "Please provide exactly one of either velocity3D or precomputed flow!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair "
                "or precomputed 3D covariance!"
            )

        if velocity3D is None:
            velocity3D = torch.Tensor([])
        if flow_precomp is None:
            flow_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        return rasterize_flow(
            means3D,
            means2D,
            velocity3D,
            flow_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
