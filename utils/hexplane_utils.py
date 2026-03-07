"""
Multi-resolution HexPlane deformation network, inspired by 4D Gaussian Splatting.

Architecture:
  6 learnable 2D feature planes for each (i, j) pair in {X, Y, Z, T}:
      XY, XZ, XT, YZ, YT, ZT
  at multiple spatial/temporal resolutions.
  Features are sampled via bilinear interpolation, fused across planes,
  then decoded by a lightweight MLP into (d_xyz, d_rotation, d_scaling).
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  HexPlane Feature Field
# ──────────────────────────────────────────────────────────────────────

# Indices for 6 planes: (dim_i, dim_j) pairs over (X=0, Y=1, Z=2, T=3)
_PLANE_PAIRS: List[Tuple[int, int]] = [
    (0, 1),  # XY
    (0, 2),  # XZ
    (0, 3),  # XT
    (1, 2),  # YZ
    (1, 3),  # YT
    (2, 3),  # ZT
]


class HexPlaneField(nn.Module):
    """Multi-resolution HexPlane feature grids.

    For each resolution level *r*, maintains 6 learnable 2-D feature maps
    (one per coordinate-pair) of shape ``(1, C, H_r, W_r)``.

    Parameters
    ----------
    spatial_resolutions : list[int]
        Grid resolutions for spatial dimensions, e.g. ``[64, 128, 256]``.
    time_resolutions : list[int]
        Corresponding grid resolutions for the time axis.
        Must be the same length as *spatial_resolutions*.
    feat_dim : int
        Feature channels *C* per plane per resolution level.
    fusion : {"concat", "product"}
        How to fuse features from the 6 planes at each resolution level.
        * ``"concat"`` — concatenate all 6 plane features (output 6·C per level).
        * ``"product"`` — sum of element-wise products of complementary
          space–time pairs (output C per level).
    init_scale : float
        Standard deviation of the normal init for plane parameters.
    """

    FUSION_MODES = ("concat", "product")

    def __init__(
        self,
        spatial_resolutions: Sequence[int] = (64, 128),
        time_resolutions: Sequence[int] = (64, 128),
        feat_dim: int = 8,
        fusion: Literal["concat", "product"] = "product",
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(spatial_resolutions) == len(time_resolutions), (
            "spatial_resolutions and time_resolutions must have the same length"
        )
        assert fusion in self.FUSION_MODES, f"fusion must be one of {self.FUSION_MODES}"

        self.spatial_resolutions = list(spatial_resolutions)
        self.time_resolutions = list(time_resolutions)
        self.num_levels = len(spatial_resolutions)
        self.feat_dim = feat_dim
        self.fusion = fusion

        # Build planes for each resolution level
        # self.planes[level][plane_idx] → (1, C, H, W)
        self.planes = nn.ModuleList()
        for lvl in range(self.num_levels):
            s_res = spatial_resolutions[lvl]
            t_res = time_resolutions[lvl]
            # Determine resolution per axis: X,Y,Z → s_res; T → t_res
            axis_res = [s_res, s_res, s_res, t_res]  # indexed by dim 0..3
            level_planes = nn.ParameterList()
            for di, dj in _PLANE_PAIRS:
                h, w = axis_res[di], axis_res[dj]
                param = nn.Parameter(init_scale * torch.randn(1, feat_dim, h, w))
                level_planes.append(param)
            self.planes.append(level_planes)

        # Pre-compute output feature dimension
        if fusion == "concat":
            self._out_dim = self.num_levels * 6 * feat_dim
        else:  # product
            # 3 complementary pairs → C each, concatenated across levels
            self._out_dim = self.num_levels * feat_dim

    @property
    def out_dim(self) -> int:
        """Total output feature dimension for a single query point."""
        return self._out_dim

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        """Sample and fuse plane features.

        Parameters
        ----------
        xyzt : Tensor of shape ``(N, 4)``
            Normalised coordinates in [-1, 1] for each axis.

        Returns
        -------
        Tensor of shape ``(N, out_dim)``
        """
        N = xyzt.shape[0]
        level_feats: list[torch.Tensor] = []

        for lvl in range(self.num_levels):
            plane_feats: list[torch.Tensor] = []
            for pidx, (di, dj) in enumerate(_PLANE_PAIRS):
                # Build 2-D sample grid: (1, N, 1, 2) for grid_sample
                # grid_sample expects (x, y) with x → W dim, y → H dim
                coords = torch.stack([xyzt[:, dj], xyzt[:, di]], dim=-1)  # (N, 2)
                grid = coords.view(1, N, 1, 2)  # (1, N, 1, 2)
                feat = F.grid_sample(
                    self.planes[lvl][pidx],  # (1, C, H, W)
                    grid,
                    align_corners=True,
                    mode="bilinear",
                    padding_mode="border",
                )  # (1, C, N, 1)
                plane_feats.append(feat.squeeze(0).squeeze(-1).T)  # (N, C)

            if self.fusion == "concat":
                level_feats.append(torch.cat(plane_feats, dim=-1))  # (N, 6C)
            else:
                # Product fusion: 3 complementary space–time pairs
                # (XY, ZT), (XZ, YT), (XZ, ZT) → but canonical pairings:
                # pair 0: XY(0) ⊙ ZT(5)
                # pair 1: XZ(1) ⊙ YT(4)
                # pair 2: XT(2) ⊙ YZ(3)
                fused = (
                    plane_feats[0] * plane_feats[5]   # XY ⊙ ZT
                    + plane_feats[1] * plane_feats[4]  # XZ ⊙ YT
                    + plane_feats[2] * plane_feats[3]  # XT ⊙ YZ
                )  # (N, C)
                level_feats.append(fused)

        return torch.cat(level_feats, dim=-1)  # (N, out_dim)

    # ── Regularisation helpers ────────────────────────────────────────

    def compute_plane_tv(self) -> torch.Tensor:
        """Total Variation loss on all planes (L1)."""
        tv = torch.tensor(0.0, device=next(self.parameters()).device)
        for lvl_planes in self.planes:
            for p in lvl_planes:
                # p: (1, C, H, W)
                tv = tv + (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
                tv = tv + (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
        return tv

    def compute_plane_l1(self) -> torch.Tensor:
        """L1 sparsity loss on all plane parameters."""
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for lvl_planes in self.planes:
            for p in lvl_planes:
                l1 = l1 + p.abs().mean()
        return l1


# ──────────────────────────────────────────────────────────────────────
#  MLP Decoder
# ──────────────────────────────────────────────────────────────────────

class HexPlaneMLPDecoder(nn.Module):
    """Lightweight MLP that maps fused HexPlane features to deformation outputs.

    Output layout: ``(d_xyz: 3, d_rotation: 4, d_scaling: 3)`` = 10 dims.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        out_dim: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Zero-init last layer so deformation starts near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
#  Composite Network: HexPlane + MLP Decoder
# ──────────────────────────────────────────────────────────────────────

class HexPlaneDeformNetwork(nn.Module):
    """4DGS-style deformation network: multi-resolution HexPlane → small MLP.

    Accepts raw (xyz, t) inputs, normalises them using a learnable or fixed
    AABB, queries the HexPlane field, and decodes deformation outputs.

    Parameters
    ----------
    spatial_resolutions : list[int]
        Multi-resolution grid sizes for spatial axes.
    time_resolutions : list[int]
        Multi-resolution grid sizes for the temporal axis.
    feat_dim : int
        Feature channels per plane per resolution level.
    mlp_hidden_dim : int
        Hidden layer width for the MLP decoder.
    mlp_num_hidden : int
        Number of hidden layers in the MLP decoder.
    fusion : str
        HexPlane fusion strategy: ``"concat"`` or ``"product"``.
    init_scale : float
        Plane parameter initialisation scale.
    is_6dof : bool
        If True, output uses SE(3) representation (6-DoF) instead of direct
        (d_xyz, d_rotation, d_scaling). *Currently not implemented — falls
        back to direct output.*
    """

    def __init__(
        self,
        spatial_resolutions: Sequence[int] = (64, 128),
        time_resolutions: Sequence[int] = (64, 128),
        feat_dim: int = 8,
        mlp_hidden_dim: int = 64,
        mlp_num_hidden: int = 2,
        fusion: Literal["concat", "product"] = "product",
        init_scale: float = 0.1,
        is_6dof: bool = False,
    ) -> None:
        super().__init__()
        self.is_6dof = is_6dof

        self.hexplane = HexPlaneField(
            spatial_resolutions=spatial_resolutions,
            time_resolutions=time_resolutions,
            feat_dim=feat_dim,
            fusion=fusion,
            init_scale=init_scale,
        )

        # Output: (d_xyz: 3, d_rotation: 4, d_scaling: 3) = 10
        out_dim = 10
        self.decoder = HexPlaneMLPDecoder(
            in_dim=self.hexplane.out_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden,
            out_dim=out_dim,
        )

        # AABB for normalising xyz to [-1, 1]. Updated via set_aabb().
        # Default: unit cube centered at origin
        self.register_buffer("aabb_min", torch.tensor([-1.0, -1.0, -1.0]))
        self.register_buffer("aabb_max", torch.tensor([1.0, 1.0, 1.0]))

    # ---- AABB management -------------------------------------------------

    def set_aabb(self, points: torch.Tensor, padding: float = 0.1) -> None:
        """Compute and store AABB from a point cloud.

        Parameters
        ----------
        points : Tensor of shape ``(N, 3)``
        padding : float
            Fractional padding added around the tight AABB.
        """
        pmin = points.min(dim=0).values
        pmax = points.max(dim=0).values
        extent = (pmax - pmin).clamp(min=1e-6)
        self.aabb_min.copy_(pmin - padding * extent)
        self.aabb_max.copy_(pmax + padding * extent)

    def _normalise_xyzt(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Normalise xyz to [-1, 1] using AABB; t is assumed already in [0, 1]
        and mapped to [-1, 1]."""
        # xyz normalisation
        xyz_norm = 2.0 * (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8) - 1.0
        # t: [0, 1] → [-1, 1]
        t_norm = 2.0 * t - 1.0
        return torch.cat([xyz_norm, t_norm], dim=-1)  # (N, 4)

    # ---- Forward ---------------------------------------------------------

    def forward(
        self, xyz: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz : Tensor ``(N, 3)`` — Gaussian centre positions.
        t   : Tensor ``(N, 1)`` — normalised time stamps in [0, 1].

        Returns
        -------
        d_xyz      : ``(N, 3)``
        d_rotation : ``(N, 4)``
        d_scaling  : ``(N, 3)``
        """
        xyzt = self._normalise_xyzt(xyz, t)       # (N, 4)
        feats = self.hexplane(xyzt)                # (N, F)
        out = self.decoder(feats)                  # (N, 10)

        d_xyz = out[:, :3]
        d_rotation = out[:, 3:7]
        d_scaling = out[:, 7:10]
        return d_xyz, d_rotation, d_scaling

    # ---- Regularisation ---------------------------------------------------

    def get_plane_tv_loss(self) -> torch.Tensor:
        return self.hexplane.compute_plane_tv()

    def get_plane_l1_loss(self) -> torch.Tensor:
        return self.hexplane.compute_plane_l1()
