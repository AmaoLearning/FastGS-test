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
from typing import Optional
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.optic_flow_utils import forward_backward_consistency_check


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, depth=None,
                 flow_fwd_path=None, flow_bwd_path=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None

        # Optical flow: 延迟加载 —— 仅存储路径，训练时按需 load_flow()
        self.flow_fwd_path: Optional[str] = flow_fwd_path
        self.flow_bwd_path: Optional[str] = flow_bwd_path
        self.flow_fwd: Optional[torch.Tensor] = None   # [2, H, W] loaded on demand
        self.flow_bwd: Optional[torch.Tensor] = None   # [2, H, W] loaded on demand
        self.flow_mask: Optional[torch.Tensor] = None   # [1, H, W] computed on demand

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(
            self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)
        # 注意：flow 由 load_flow/unload_flow 独立管理，不在此处处理

    @property
    def has_flow(self) -> bool:
        """该相机是否有光流文件可供加载。"""
        return self.flow_fwd_path is not None

    def load_flow(self, device: str = 'cuda', flow_magnitude_thresh: float = 0.0) -> None:
        """按需从磁盘加载光流到指定设备，并计算组合掩码。

        组合掩码 = 前后一致性掩码 & 模长阈值掩码：
        - 一致性掩码：排除前后向矛盾的不可靠像素
        - 模长阈值掩码：排除 |flow| < thresh 的静态噪声像素

        光流以 float16 存储以节省显存。若已加载则跳过（幂等操作）。
        """
        if self.flow_fwd is not None:
            return  # 已加载
        if self.flow_fwd_path is not None and os.path.exists(self.flow_fwd_path):
            arr = np.load(self.flow_fwd_path)            # [H, W, 2]
            self.flow_fwd = torch.from_numpy(arr).permute(2, 0, 1).to(
                dtype=torch.float16, device=device)       # [2, H, W] fp16
        if self.flow_bwd_path is not None and os.path.exists(self.flow_bwd_path):
            arr = np.load(self.flow_bwd_path)
            self.flow_bwd = torch.from_numpy(arr).permute(2, 0, 1).to(
                dtype=torch.float16, device=device)

        # ── 1. Forward-Backward Consistency Mask ──
        consistency_mask = None
        if self.flow_fwd is not None and self.flow_bwd is not None:
            mask_f32 = forward_backward_consistency_check(
                self.flow_fwd.float(), self.flow_bwd.float(),
                # 使用函数默认严格参数 alpha1=0.005, alpha2=0.2
            )  # [1, H, W] float32, 0.0/1.0
            consistency_mask = mask_f32 > 0.5  # [1, H, W] bool

        # ── 2. Magnitude Threshold Mask（抑制静态区域 RAFT 噪声）──
        mag_mask = None
        if self.flow_fwd is not None and flow_magnitude_thresh > 0:
            flow_mag = self.flow_fwd.float().norm(dim=0, keepdim=True)  # [1, H, W]
            mag_mask = flow_mag > flow_magnitude_thresh  # [1, H, W] bool

        # ── 3. Combine masks ──
        if consistency_mask is not None and mag_mask is not None:
            self.flow_mask = consistency_mask & mag_mask
        elif consistency_mask is not None:
            self.flow_mask = consistency_mask
        elif mag_mask is not None:
            self.flow_mask = mag_mask
        # else: self.flow_mask remains None → train.py will use all-ones fallback

    def unload_flow(self) -> None:
        """释放光流张量以回收 GPU/CPU 内存。"""
        self.flow_fwd = None
        self.flow_bwd = None
        self.flow_mask = None


class LazyCamera(nn.Module):
    """Memory-efficient camera: stores all geometric metadata, defers image IO.

    Compatible with :class:`Camera` as a duck-type for rendering / loss
    computation.  The image tensor is loaded on-demand via
    :meth:`load_image_to_gpu` or injected externally by the async DataLoader.
    """

    def __init__(
        self,
        colmap_id: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image_path: str,
        image_name: str,
        uid: int,
        width: int,
        height: int,
        target_resolution: tuple = None,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
        fid: float = None,
        depth: np.ndarray = None,
        flow_fwd_path: Optional[str] = None,
        flow_bwd_path: Optional[str] = None,
    ):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path

        # Target pixel resolution (w, h) — may differ from raw image on disk
        self._target_resolution: tuple = target_resolution or (width, height)
        self.image_width: int = self._target_resolution[0]
        self.image_height: int = self._target_resolution[1]

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # Image tensor — NOT loaded at __init__; injected by DataLoader
        # or via load_image_to_gpu()
        self.original_image: Optional[torch.Tensor] = None

        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None

        # Optical flow: lazy loading (identical to Camera)
        self.flow_fwd_path: Optional[str] = flow_fwd_path
        self.flow_bwd_path: Optional[str] = flow_bwd_path
        self.flow_fwd: Optional[torch.Tensor] = None
        self.flow_bwd: Optional[torch.Tensor] = None
        self.flow_mask: Optional[torch.Tensor] = None

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        # Projection matrices — identical computation to Camera
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # ── Image lifecycle ──────────────────────────────────────────────

    def load_image_to_gpu(self, device: str = "cuda") -> None:
        """Read image from disk, resize, normalise to [0,1], move to *device*.

        Idempotent: skips if ``original_image`` is already set.
        Uses ``torchvision.io.read_image`` (fast C++ decoder) with PIL fallback.
        """
        if self.original_image is not None:
            return

        target_w, target_h = self._target_resolution

        try:
            from torchvision.io import read_image as _tv_read, ImageReadMode
            img = _tv_read(self.image_path, mode=ImageReadMode.RGB)  # [3,H,W] uint8
        except Exception:
            from PIL import Image as _PILImage
            pil_img = _PILImage.open(self.image_path).convert("RGB")
            img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)  # [3,H,W]

        if img.shape[1] != target_h or img.shape[2] != target_w:
            img = (
                torch.nn.functional.interpolate(
                    img.unsqueeze(0).float(),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .clamp_(0, 255)
                .to(torch.uint8)
            )

        self.original_image = img.float().div_(255.0).clamp_(0.0, 1.0).to(device)

    def unload_image(self) -> None:
        """Release the image tensor to reclaim GPU / CPU memory."""
        self.original_image = None

    # ── Flow lifecycle (same as Camera) ──────────────────────────────

    @property
    def has_flow(self) -> bool:
        return self.flow_fwd_path is not None

    def load_flow(self, device: str = "cuda", flow_magnitude_thresh: float = 0.0) -> None:
        """同 Camera.load_flow — 一致性 + 模长组合掩码。"""
        if self.flow_fwd is not None:
            return
        if self.flow_fwd_path is not None and os.path.exists(self.flow_fwd_path):
            arr = np.load(self.flow_fwd_path)
            self.flow_fwd = torch.from_numpy(arr).permute(2, 0, 1).to(
                dtype=torch.float16, device=device)
        if self.flow_bwd_path is not None and os.path.exists(self.flow_bwd_path):
            arr = np.load(self.flow_bwd_path)
            self.flow_bwd = torch.from_numpy(arr).permute(2, 0, 1).to(
                dtype=torch.float16, device=device)

        consistency_mask = None
        if self.flow_fwd is not None and self.flow_bwd is not None:
            mask_f32 = forward_backward_consistency_check(
                self.flow_fwd.float(), self.flow_bwd.float(),
            )
            consistency_mask = mask_f32 > 0.5

        mag_mask = None
        if self.flow_fwd is not None and flow_magnitude_thresh > 0:
            flow_mag = self.flow_fwd.float().norm(dim=0, keepdim=True)
            mag_mask = flow_mag > flow_magnitude_thresh

        if consistency_mask is not None and mag_mask is not None:
            self.flow_mask = consistency_mask & mag_mask
        elif consistency_mask is not None:
            self.flow_mask = consistency_mask
        elif mag_mask is not None:
            self.flow_mask = mag_mask

    def unload_flow(self) -> None:
        self.flow_fwd = None
        self.flow_bwd = None
        self.flow_mask = None

    # ── Utility ──────────────────────────────────────────────────────

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, self.trans, self.scale)
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device: str = "cuda"):
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)
        if self.original_image is not None:
            self.original_image = self.original_image.to(data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
