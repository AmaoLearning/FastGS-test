"""Asynchronous image + optical-flow prefetch pipeline for LazyCamera training.

Decouples disk IO / JPEG-PNG decoding **and** numpy flow loading from GPU
training by leveraging PyTorch DataLoader multi-processing.  Workers read
and resize images (and optionally load flow ``.npy`` files) in the
background; ``pin_memory`` enables non-blocking CUDA transfers.

Usage (inside Scene)::

    from utils.dataload_utils import create_camera_dataloader, InfiniteDataLoader

    dl = create_camera_dataloader(lazy_train_cameras, num_workers=8, load_flow=True)
    data_iter = InfiniteDataLoader(dl)

    for iteration in range(1, max_iter + 1):
        batch = next(data_iter)
        cam_idx, image, flow_fwd, flow_bwd = batch
        ...
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

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


# ── Flow loading (runs inside DataLoader worker processes) ────────────

# Sentinel tensor returned when a camera has no flow file.
# Shape [0] — easily distinguishable from real [2, H, W] data.
_EMPTY_FLOW = torch.empty(0, dtype=torch.float16)


def _load_flow_tensor(path: Optional[str]) -> torch.Tensor:
    """Load a single flow ``.npy`` file.  Returns fp16 [2, H, W] or _EMPTY_FLOW."""
    if path is None or not os.path.exists(path):
        return _EMPTY_FLOW
    arr = np.load(path)  # expected [H, W, 2], float32
    return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float16).contiguous()


# ── Dataset (lightweight & pickle-safe for multi-process workers) ─────

class CameraDataset(Dataset):
    """Stores only file paths + resolutions — no torch tensors or nn.Modules.

    ``__getitem__`` returns ``(cam_index, image_tensor, flow_fwd, flow_bwd)``
    where *image_tensor* is a float32 [3, H, W] tensor normalised to [0, 1],
    and *flow_fwd* / *flow_bwd* are fp16 [2, H, W] tensors (or empty [0]
    tensors when unavailable).
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        target_resolutions: Sequence[Tuple[int, int]],
        flow_fwd_paths: Optional[Sequence[Optional[str]]] = None,
        flow_bwd_paths: Optional[Sequence[Optional[str]]] = None,
        load_flow: bool = False,
    ) -> None:
        assert len(image_paths) == len(target_resolutions)
        self.image_paths: List[str] = list(image_paths)
        self.target_resolutions: List[Tuple[int, int]] = list(target_resolutions)
        self.load_flow: bool = load_flow

        n = len(image_paths)
        self.flow_fwd_paths: List[Optional[str]] = (
            list(flow_fwd_paths) if flow_fwd_paths is not None else [None] * n
        )
        self.flow_bwd_paths: List[Optional[str]] = (
            list(flow_bwd_paths) if flow_bwd_paths is not None else [None] * n
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        w, h = self.target_resolutions[idx]
        img = _load_image_tensor(self.image_paths[idx], w, h)

        if self.load_flow:
            flow_fwd = _load_flow_tensor(self.flow_fwd_paths[idx])
            flow_bwd = _load_flow_tensor(self.flow_bwd_paths[idx])
        else:
            flow_fwd = _EMPTY_FLOW
            flow_bwd = _EMPTY_FLOW

        return idx, img, flow_fwd, flow_bwd


# ── Custom collate (handles variable-shape flow tensors) ──────────────

def _collate_camera_batch(
    batch: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for batch_size=1.  Stacks idx & image; passes flow through."""
    assert len(batch) == 1, "CameraDataset is designed for batch_size=1"
    idx, img, flow_fwd, flow_bwd = batch[0]
    return torch.tensor(idx, dtype=torch.long), img.unsqueeze(0), flow_fwd, flow_bwd


# ── DataLoader factory ────────────────────────────────────────────────

def create_camera_dataloader(
    cameras,
    batch_size: int = 1,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = True,
    load_flow: bool = False,
) -> DataLoader:
    """Build an optimised :class:`DataLoader` from a list of *LazyCamera* objects.

    Only serialisable paths / scalars are forwarded to worker processes.

    Args:
        cameras: list of ``LazyCamera``.
        load_flow: if True, workers also read ``flow_fwd_path`` / ``flow_bwd_path``
            numpy arrays in parallel with image loading.
    """
    image_paths = [c.image_path for c in cameras]
    target_resolutions = [c._target_resolution for c in cameras]

    flow_fwd_paths: Optional[List[Optional[str]]] = None
    flow_bwd_paths: Optional[List[Optional[str]]] = None
    if load_flow:
        flow_fwd_paths = [getattr(c, 'flow_fwd_path', None) for c in cameras]
        flow_bwd_paths = [getattr(c, 'flow_bwd_path', None) for c in cameras]

    dataset = CameraDataset(
        image_paths, target_resolutions,
        flow_fwd_paths=flow_fwd_paths,
        flow_bwd_paths=flow_bwd_paths,
        load_flow=load_flow,
    )

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
        collate_fn=_collate_camera_batch,
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
