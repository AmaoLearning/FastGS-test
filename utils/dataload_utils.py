"""Asynchronous image prefetch pipeline for LazyCamera training.

Decouples disk IO / JPEG-PNG decoding from GPU training by leveraging
PyTorch DataLoader multi-processing.  Workers read and resize images in
the background; ``pin_memory`` enables non-blocking CUDA transfers.

Usage (inside train.py)::

    from utils.dataloader import create_camera_dataloader, InfiniteDataLoader

    dl = create_camera_dataloader(lazy_train_cameras, num_workers=8)
    data_iter = InfiniteDataLoader(dl)

    for iteration in range(1, max_iter + 1):
        cam_idx_t, image_t = next(data_iter)          # prefetched & pinned
        viewpoint_cam = lazy_train_cameras[cam_idx_t.item()]
        viewpoint_cam.original_image = image_t.squeeze(0).to("cuda", non_blocking=True)
        ...
        viewpoint_cam.original_image = None            # free VRAM immediately
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Fast C++ image decoder (torchvision >= 0.8) ───────────────────────
try:
    from torchvision.io import read_image as _tv_read_image, ImageReadMode

    _HAS_TV_IO = True
except ImportError:
    _HAS_TV_IO = False


# ── Image loading (runs inside DataLoader worker processes) ───────────

def _pil_fallback(path: str) -> torch.Tensor:
    """Load an image via PIL — universal fallback.

    Returns:
        [3, H, W] uint8 tensor.
    """
    from PIL import Image

    pil_img = Image.open(path).convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_image_tensor(path: str, target_w: int, target_h: int) -> torch.Tensor:
    """Read *path*, resize to (*target_w*, *target_h*), return float32 [C,H,W] in [0,1].

    Prefers ``torchvision.io.read_image`` (C++ JPEG/PNG decoder) for speed;
    falls back to PIL if unavailable or if the file format is unsupported.
    """
    if _HAS_TV_IO:
        try:
            img = _tv_read_image(path, mode=ImageReadMode.RGB)  # [3,H,W] uint8
        except Exception:
            img = _pil_fallback(path)
    else:
        img = _pil_fallback(path)

    # Resize when on-disk resolution differs from target
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

    return img.float().div_(255.0)


# ── Dataset (lightweight & pickle-safe for multi-process workers) ─────

class CameraDataset(Dataset):
    """Stores only file paths + resolutions — no torch tensors or nn.Modules.

    ``__getitem__`` returns ``(cam_index, image_tensor)`` where
    *image_tensor* is a float32 [3, H, W] tensor normalised to [0, 1].
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        target_resolutions: Sequence[Tuple[int, int]],
    ) -> None:
        assert len(image_paths) == len(target_resolutions)
        self.image_paths: List[str] = list(image_paths)
        self.target_resolutions: List[Tuple[int, int]] = list(target_resolutions)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        w, h = self.target_resolutions[idx]
        img = _load_image_tensor(self.image_paths[idx], w, h)
        return idx, img


# ── DataLoader factory ────────────────────────────────────────────────

def create_camera_dataloader(
    cameras,
    batch_size: int = 1,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Build an optimised :class:`DataLoader` from a list of *LazyCamera* objects.

    Only ``image_path`` and ``_target_resolution`` are extracted and forwarded
    to worker processes (lightweight, pickle-safe).

    Args:
        cameras: list of ``LazyCamera`` (only ``.image_path`` and
            ``._target_resolution`` are read).
        num_workers: IO / decode parallelism.  8 is a good default for NVMe.
        prefetch_factor: how many batches each worker pre-loads.
        pin_memory: enables ``non_blocking=True`` CUDA transfers.
    """
    image_paths = [c.image_path for c in cameras]
    target_resolutions = [c._target_resolution for c in cameras]
    dataset = CameraDataset(image_paths, target_resolutions)

    actual_workers = min(num_workers, max(1, len(cameras)))
    if actual_workers <= 0:
        actual_workers = 0
        prefetch_factor = None
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=actual_workers,
        pin_memory=pin_memory and (actual_workers > 0),
        prefetch_factor=prefetch_factor if actual_workers > 0 else None,
        persistent_workers=persistent_workers and (actual_workers > 0),
        drop_last=False,
    )


# ── Infinite iterator (training runs for N iterations, not epochs) ────

class InfiniteDataLoader:
    """Wrap a :class:`DataLoader` to yield batches endlessly (auto re-shuffle)."""

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)
