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
from scene.deform_model import DeformModel, DeformModel_4DGS, ClusteredDeformModel
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
        self._lazy_mode: bool = args.lazy_load
        self._load2gpu_on_the_fly: bool = args.load2gpu_on_the_fly
        self._lazy_data_iter = None
        self._gpu_image_buffers: Optional[List[torch.Tensor]] = None
        self._lazy_cameras: Optional[List] = None
        self._active_buf_idx: int = 0
        self._dma_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_ready: Optional[Tuple[int, int, torch.cuda.Event]] = None
        self._viewpoint_stack: list = []  # for non-lazy random sampling

        self._lazy_num_workers: int = int(args.lazy_num_workers)
        self._lazy_prefetch_factor: int = int(args.lazy_prefetch_factor)
        self._lazy_image_buffer_count: int = max(2, int(args.lazy_image_buffer_count))
        self._lazy_prefetch_flow_to_cache: bool = bool(args.lazy_prefetch_flow_to_cache)
        self._enable_flow_preload_cache: bool = bool(args.enable_flow_preload_cache)
        self._flow_preload_cache_size: int = int(args.flow_preload_cache_size)
        self._flow_preload_cache_device: str = str(args.flow_preload_cache_device)

        load_flow = args.use_flow_loss or args.use_flow_mask or args.use_dynamic_sep
        self._load_flow: bool = load_flow
        self._load_depth: bool = False

        # Number of *temporal* frames (used for time_interval in velocity
        # loss).  For multi-camera datasets the total camera count is
        # spatial × temporal; only the temporal part should define dt.
        self._num_temporal_frames: Optional[int] = None

        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            if args.lazy_load:
                _n_frames = args.num_images
                self._num_temporal_frames = _n_frames
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

        # Fallback: if _num_temporal_frames was not set by a specific loader,
        # assume each camera corresponds to one unique time step.
        if self._num_temporal_frames is None:
            self._num_temporal_frames = len(
                self.train_cameras.get(resolution_scales[0], []))
            print(f"[Scene] num_temporal_frames = {self._num_temporal_frames} "
                  f"(fallback: len(train_cameras))")
        else:
            print(f"[Scene] num_temporal_frames = {self._num_temporal_frames} "
                  f"(total train cameras = "
                  f"{len(self.train_cameras.get(resolution_scales[0], []))})")

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
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        scale: float = 1.0,
    ) -> None:
        """Initialize the async image prefetch pipeline.

        Call once after ``__init__``.  No-op if ``lazy_load`` is disabled.
        When all cameras share the same resolution, a persistent GPU buffer
        is pre-allocated so that training iterations incur **zero**
        ``cudaMalloc`` / ``cudaFree``.

        Flow prefetching is optional. When ``lazy_prefetch_flow_to_cache`` is
        enabled and flow cache is enabled, DataLoader workers also load flow
        files and the main process pre-warms the flow cache in the same camera
        order as images.
        """
        from utils.dataload_utils import create_camera_dataloader, InfiniteDataLoader

        if num_workers is None:
            num_workers = self._lazy_num_workers
        if prefetch_factor is None:
            prefetch_factor = self._lazy_prefetch_factor

        cameras = self.getTrainCameras(scale)
        self._lazy_cameras = cameras
        _prefetch_flow = (
            self._lazy_prefetch_flow_to_cache
            and self._load_flow
            and self._enable_flow_preload_cache
            and self._flow_preload_cache_size > 0
        )
        _dl = create_camera_dataloader(
            cameras,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            load_flow=_prefetch_flow,
            load_depth=self._load_depth,
        )
        self._lazy_data_iter = InfiniteDataLoader(_dl)

        # Pre-allocate a fixed CUDA buffer if every camera uses the same
        # resolution (the common case for N3V).  This avoids even the
        # caching-allocator look-up on every iteration.
        resolutions = set(c._target_resolution for c in cameras)
        if len(resolutions) == 1:
            w, h = resolutions.pop()
            self._gpu_image_buffers = [
                torch.empty(3, h, w, dtype=torch.float32, device="cuda")
                for _ in range(self._lazy_image_buffer_count)
            ]
            self._dma_stream = torch.cuda.Stream()
            # Pre-fill: DMA the first batch so the first
            # next_train_camera() call never stalls on a cold buffer.
            self._kick_prefetch()
            n_with_flow = sum(1 for c in cameras if c.has_flow)
            if n_with_flow > 0:
                if _prefetch_flow:
                    flow_info = f", flow_files={n_with_flow}/{len(cameras)} (prefetch->cache:{self._flow_preload_cache_device}, size={self._flow_preload_cache_size})"
                else:
                    flow_info = f", flow_files={n_with_flow}/{len(cameras)} (on-demand)"
            else:
                flow_info = ""
            print(f"[INFO] Lazy DataLoader: {len(cameras)} cameras, "
                  f"{num_workers} workers, prefetch={prefetch_factor}, "
                  f"GPU image-buffers={len(self._gpu_image_buffers)}x{w}x{h} (zero-alloc, DMA overlap){flow_info}")
        else:
            print(f"[INFO] Lazy DataLoader: {len(cameras)} cameras, "
                  f"{num_workers} workers, prefetch={prefetch_factor}, "
                  f"(mixed resolutions, no persistent buffer)")

    def _kick_prefetch(
        self, wait_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Start async DMA of the next DataLoader batch into the inactive
        GPU buffer on ``self._dma_stream``.

        Called at the END of ``next_train_camera()`` so the transfer runs
        concurrently with render + backward on the default stream.

        Args:
            wait_event: If given, the DMA stream first waits for this
                event (captures previous compute on the target buffer)
                to avoid a WAR hazard.
        """
        _batch_idx, _batch_img, _batch_flow_fwd, _batch_flow_bwd, _batch_depth, _batch_depth_conf = next(self._lazy_data_iter)
        cam_idx = _batch_idx.item()
        pinned_img = _batch_img.squeeze(0)          # [3,H,W] pinned CPU

        if (self._lazy_prefetch_flow_to_cache
                and self._enable_flow_preload_cache
                and self._flow_preload_cache_size > 0
                and self._lazy_cameras is not None
                and (_batch_flow_fwd.numel() > 0 or _batch_flow_bwd.numel() > 0)):
            from scene.cameras import preload_flow_cache_from_tensors
            _cam = self._lazy_cameras[cam_idx]
            preload_flow_cache_from_tensors(
                _cam.flow_fwd_path,
                _cam.flow_bwd_path,
                _batch_flow_fwd,
                _batch_flow_bwd,
                cache_device=self._flow_preload_cache_device,
                cache_size=self._flow_preload_cache_size,
            )

        # Pre-assign depth tensors from DataLoader to the camera so that
        # load_depth() in train.py can move them to GPU without re-reading
        # from disk.
        if (self._load_depth
                and self._lazy_cameras is not None
                and (_batch_depth.numel() > 0 or _batch_depth_conf.numel() > 0)):
            _cam = self._lazy_cameras[cam_idx]
            _cam._prefetched_depth = _batch_depth
            _cam._prefetched_depth_conf = _batch_depth_conf

        next_buf = (self._active_buf_idx + 1) % len(self._gpu_image_buffers)
        with torch.cuda.stream(self._dma_stream):
            if wait_event is not None:
                self._dma_stream.wait_event(wait_event)
            self._gpu_image_buffers[next_buf].copy_(
                pinned_img, non_blocking=True)

        self._prefetch_ready = (
            cam_idx,
            next_buf,
            self._dma_stream.record_event(),
        )

    def next_train_camera(self, scale: float = 1.0):
        """Return ``(camera, total_frame)`` with the image ready on GPU.

        * **Lazy mode** — pulls from the async DataLoader and copies the
          pre-fetched pinned tensor into the persistent GPU buffer (or
          falls back to ``.to('cuda')`` for mixed-resolution datasets).
          Flow tensors can also be optionally pre-warmed into cache here
          (same DataLoader order), while the actual ``Camera.load_flow()``
          in ``train.py`` keeps ownership of per-iteration flow tensors.
        * **Eager mode** — pops a random camera from an internal
          ``viewpoint_stack`` that auto-refills each epoch.
        """
        if self._lazy_mode:
            assert self._lazy_data_iter is not None, (
                "Call scene.setup_lazy_dataloader() before next_train_camera()")
            cameras = self.getTrainCameras(scale)

            if (self._gpu_image_buffers is not None
                    and self._prefetch_ready is not None):
                # ── Double-buffer fast path ──────────────────────────
                # Record that all previous default-stream work (render,
                # backward, optimizer) from the last iteration is done.
                # The DMA stream will wait on this before writing into
                # the buffer that was just freed.
                compute_done = torch.cuda.current_stream().record_event()

                cam_idx, buf_idx, dma_event = self._prefetch_ready

                # Wait for the prefetched DMA to finish — near-instant
                # since it started during the previous iteration's compute.
                torch.cuda.current_stream().wait_event(dma_event)

                self._active_buf_idx = buf_idx
                cam = cameras[cam_idx]
                cam.original_image = self._gpu_image_buffers[buf_idx]

                # Kick prefetch for the NEXT iteration.  The DMA runs on
                # self._dma_stream and fully overlaps with upcoming
                # render + backward on the default stream.
                self._kick_prefetch(compute_done)

                return cam, self._num_temporal_frames
            else:
                # Fallback (mixed resolutions): no double buffer
                _batch_idx, _batch_img, _, _, _, _ = next(self._lazy_data_iter)
                cam = cameras[_batch_idx.item()]
                cam.original_image = _batch_img.squeeze(0).to(
                    "cuda", non_blocking=True)
                return cam, self._num_temporal_frames

        # ── Eager (non-lazy) path ──
        if not self._viewpoint_stack:
            self._viewpoint_stack = self.getTrainCameras(scale).copy()

        idx = random.randint(0, len(self._viewpoint_stack) - 1)
        cam = self._viewpoint_stack.pop(idx)

        if self._load2gpu_on_the_fly:
            cam.load2device()

        return cam, self._num_temporal_frames

    def release_camera_image(self, cam) -> None:
        """Drop the camera's image reference after ``loss.backward()``.

        In lazy-buffer mode the underlying GPU memory stays allocated on
        ``self._gpu_image_buffers`` — only the Python reference is cleared,
        so no ``cudaFree`` ever occurs.

        Flow data is managed separately via ``unload_flow()`` in train.py.
        """
        if self._lazy_mode:
            cam.original_image = None          # buffer stays alive on self
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
