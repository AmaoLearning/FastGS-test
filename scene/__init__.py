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
import random
import json
from typing import Optional, List, Tuple

import torch

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # ── Lazy loading state ──
        self._lazy_mode: bool = getattr(args, 'lazy_load', False)
        self._load2gpu_on_the_fly: bool = getattr(args, 'load2gpu_on_the_fly', False)
        self._lazy_data_iter = None
        self._gpu_image_buffer: Optional[torch.Tensor] = None
        self._gpu_flow_fwd_buffer: Optional[torch.Tensor] = None
        self._gpu_flow_bwd_buffer: Optional[torch.Tensor] = None
        self._viewpoint_stack: list = []  # for non-lazy random sampling

        load_flow = getattr(args, 'use_flow_loss', False)
        self._load_flow: bool = load_flow

        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            if getattr(args, 'lazy_load', False):
                _n_frames = getattr(args, 'num_images', 300)
                print(f"Found poses_bounds.npy, using N3V lazy loader "
                      f"(metadata only, {_n_frames} frames/cam, flow={load_flow})")
                scene_info = sceneLoadTypeCallbacks["N3VLazy"](
                    args.source_path, args.white_background, args.eval,
                    num_frames=_n_frames, load_flow=load_flow)
            else:
                # print(f"Found poses_bounds.npy, assuming Neu3D/LLFF data set! (num_t=300, flow={load_flow})")
                # scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, num_images=30,
                #                                                        load_flow=load_flow)
                print("Found poses_bounds.npy file, assuming Neur3D data set!")
                scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          load_flow=load_flow)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # 额外保存高斯数量到文件
        num_gaussians = self.gaussians.get_xyz.shape[0]
        with open(os.path.join(point_cloud_path, "num_gaussians.txt"), 'w') as f:
            f.write(f"{num_gaussians}\n")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    # ── Lazy loading API ─────────────────────────────────────────────

    @property
    def lazy_mode(self) -> bool:
        return self._lazy_mode

    def setup_lazy_dataloader(
        self,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        scale: float = 1.0,
    ) -> None:
        """Initialize the async image (+ optional flow) prefetch pipeline.

        Call once after ``__init__``.  No-op if ``lazy_load`` is disabled.
        When all cameras share the same resolution, persistent GPU buffers
        are pre-allocated so that training iterations incur **zero**
        ``cudaMalloc`` / ``cudaFree``.
        """
        from utils.dataload_utils import create_camera_dataloader, InfiniteDataLoader

        cameras = self.getTrainCameras(scale)
        _dl = create_camera_dataloader(
            cameras,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            load_flow=self._load_flow,
        )
        self._lazy_data_iter = InfiniteDataLoader(_dl)

        # Pre-allocate a fixed CUDA buffer if every camera uses the same
        # resolution (the common case for N3V).  This avoids even the
        # caching-allocator look-up on every iteration.
        resolutions = set(c._target_resolution for c in cameras)
        if len(resolutions) == 1:
            w, h = resolutions.pop()
            self._gpu_image_buffer = torch.empty(
                3, h, w, dtype=torch.float32, device="cuda")

            # Flow buffers: fp16 [2, H, W] — allocated only if flow is used.
            if self._load_flow:
                self._gpu_flow_fwd_buffer = torch.empty(
                    2, h, w, dtype=torch.float16, device="cuda")
                self._gpu_flow_bwd_buffer = torch.empty(
                    2, h, w, dtype=torch.float16, device="cuda")

            flow_info = ""
            if self._load_flow:
                n_with_flow = sum(1 for c in cameras if c.has_flow)
                flow_info = f", flow={n_with_flow}/{len(cameras)}"
            print(f"[INFO] Lazy DataLoader: {len(cameras)} cameras, "
                  f"{num_workers} workers, prefetch={prefetch_factor}, "
                  f"GPU buffer={w}x{h} (zero-alloc){flow_info}")
        else:
            print(f"[INFO] Lazy DataLoader: {len(cameras)} cameras, "
                  f"{num_workers} workers, prefetch={prefetch_factor}, "
                  f"(mixed resolutions, no persistent buffer)")

    def next_train_camera(self, scale: float = 1.0):
        """Return ``(camera, total_frame)`` with the image ready on GPU.

        * **Lazy mode** — pulls from the async DataLoader and copies the
          pre-fetched pinned tensor into the persistent GPU buffer (or
          falls back to ``.to('cuda')`` for mixed-resolution datasets).
          If ``load_flow`` is enabled, prefetched flow tensors are also
          injected onto the camera and the consistency mask is computed
          on GPU (zero host-side IO stall).
        * **Eager mode** — pops a random camera from an internal
          ``viewpoint_stack`` that auto-refills each epoch.
        """
        if self._lazy_mode:
            assert self._lazy_data_iter is not None, (
                "Call scene.setup_lazy_dataloader() before next_train_camera()")
            cameras = self.getTrainCameras(scale)
            _batch_idx, _batch_img, _flow_fwd, _flow_bwd = next(self._lazy_data_iter)
            cam = cameras[_batch_idx.item()]
            pinned_img = _batch_img.squeeze(0)          # [3,H,W] pinned CPU

            if (self._gpu_image_buffer is not None
                    and pinned_img.shape == self._gpu_image_buffer.shape):
                # Zero-alloc fast path: DMA into persistent buffer
                self._gpu_image_buffer.copy_(pinned_img, non_blocking=True)
                cam.original_image = self._gpu_image_buffer
            else:
                # Fallback: caching allocator handles de/allocation
                cam.original_image = pinned_img.to("cuda", non_blocking=True)

            # ── Inject prefetched flow data ────────────────────────────
            if self._load_flow and _flow_fwd.numel() > 0:
                # _flow_fwd: [2, H, W] fp16 pinned; _flow_bwd: same
                if (self._gpu_flow_fwd_buffer is not None
                        and _flow_fwd.shape == self._gpu_flow_fwd_buffer.shape):
                    self._gpu_flow_fwd_buffer.copy_(_flow_fwd, non_blocking=True)
                    cam.flow_fwd = self._gpu_flow_fwd_buffer
                else:
                    cam.flow_fwd = _flow_fwd.to("cuda", non_blocking=True)

                if _flow_bwd.numel() > 0:
                    if (self._gpu_flow_bwd_buffer is not None
                            and _flow_bwd.shape == self._gpu_flow_bwd_buffer.shape):
                        self._gpu_flow_bwd_buffer.copy_(_flow_bwd, non_blocking=True)
                        cam.flow_bwd = self._gpu_flow_bwd_buffer
                    else:
                        cam.flow_bwd = _flow_bwd.to("cuda", non_blocking=True)

                    # Compute consistency mask on GPU (fast, avoids CPU↔GPU
                    # round-trip that the old per-iteration load_flow() did)
                    from utils.optic_flow_utils import forward_backward_consistency_check
                    mask_f32 = forward_backward_consistency_check(
                        cam.flow_fwd.float(), cam.flow_bwd.float(),
                        alpha1=0.01, alpha2=0.5,
                    )
                    cam.flow_mask = mask_f32.bool()
                else:
                    cam.flow_bwd = None
                    cam.flow_mask = None

            return cam, len(cameras)

        # ── Eager (non-lazy) path ──
        if not self._viewpoint_stack:
            self._viewpoint_stack = self.getTrainCameras(scale).copy()

        total_frame = len(self._viewpoint_stack)
        idx = random.randint(0, len(self._viewpoint_stack) - 1)
        cam = self._viewpoint_stack.pop(idx)

        if self._load2gpu_on_the_fly:
            cam.load2device()

        return cam, total_frame

    def release_camera_image(self, cam) -> None:
        """Drop the camera's image (and flow) references after ``loss.backward()``.

        In lazy-buffer mode the underlying GPU memory stays allocated on
        ``self._gpu_image_buffer`` / ``self._gpu_flow_*_buffer`` — only the
        Python references are cleared, so no ``cudaFree`` ever occurs.
        """
        if self._lazy_mode:
            cam.original_image = None          # buffer stays alive on self
            # Also clear flow references (buffers persist on Scene)
            cam.flow_fwd = None
            cam.flow_bwd = None
            cam.flow_mask = None
        elif self._load2gpu_on_the_fly:
            cam.load2device("cpu")

    def ensure_cameras_loaded(self, cameras: list) -> None:
        """Synchronously load images for a batch of cameras.

        Called before multi-view scoring (densification / pruning).
        No-op in eager mode (images are already resident).
        """
        if self._lazy_mode:
            for cam in cameras:
                cam.load_image_to_gpu("cuda")

    def release_cameras(self, cameras: list) -> None:
        """Release images for a batch of cameras after scoring.

        No-op in eager mode.
        """
        if self._lazy_mode:
            for cam in cameras:
                cam.unload_image()
