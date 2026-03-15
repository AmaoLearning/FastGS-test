"""Lightweight 2D convolutional classifier for dynamic probability map refinement.

Inspired by SA4D (Segment Any 4D Gaussians): takes a rendered (1, H, W)
probability map and outputs (1, H, W) classification logits via learned 2D
convolutions, enabling the model to capture spatial patterns that per-Gaussian
logits alone cannot express.

Training pipeline:
    prob_map  →  DynamicClassifier2D  →  logits  →  BCE(sigmoid, binarised_flow)
"""

import os
import torch
import torch.nn as nn
from typing import List


class DynamicClassifier2D(nn.Module):
    """2D CNN classifier for refining rendered dynamic probability maps.

    Architecture (large-to-small aggregation)::

        Conv7×7 → Conv5×5 → Conv3×3 → Conv1×1

    Earlier layers use large kernels to aggregate broad spatial context
    (neighborhood features), then progressively narrow the receptive field
    to refine local details. Kernel sizes are computed automatically from
    ``num_layers``: starting at ``2*num_layers - 1`` and decreasing by 2
    per layer (minimum 3), with a final 1×1 head.

    Input:  ``(1, H, W)`` probability map from Gaussian splatting.
    Output: ``(1, H, W)`` classification logits (pre-sigmoid).
    """

    def __init__(self, hidden_channels: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        # Build kernel sizes: large → small, e.g. num_layers=3 → [7, 5, 3]
        # num_layers=4 → [9, 7, 5, 3], num_layers=2 → [5, 3]
        kernel_sizes = [max(3, 2 * num_layers - 1 - 2 * i) for i in range(num_layers - 1)]

        layers: List[nn.Module] = []
        in_ch = 1
        for ks in kernel_sizes:
            pad = ks // 2
            layers.extend([
                nn.Conv2d(in_ch, hidden_channels, kernel_size=ks, padding=pad, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
            in_ch = hidden_channels
        # Final 1×1 conv → single-channel logits (no activation)
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, prob_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prob_map: ``(1, H, W)`` rendered probability map.

        Returns:
            ``(1, H, W)`` classification logits (pre-sigmoid).
        """
        x = prob_map.unsqueeze(0)   # (1, 1, H, W)
        out = self.net(x)
        return out.squeeze(0)       # (1, H, W)

    # ------------------------------------------------------------------ I/O
    def save_weights(self, model_path: str, iteration: int) -> None:
        out_dir = os.path.join(model_path, "classifier")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"classifier_{iteration}.pth")
        torch.save(self.state_dict(), path)

    def load_weights(self, model_path: str, iteration: int) -> None:
        path = os.path.join(model_path, "classifier", f"classifier_{iteration}.pth")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
