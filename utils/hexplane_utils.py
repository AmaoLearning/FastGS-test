"""
Multi-resolution HexPlane deformation network — 4DGS paper reference config.

Architecture (matching Wu et al., "4D Gaussian Splatting", CVPR 2024):
  1. Positional Encoding of raw xyz (multires=10 → 63-d) and t (multires=6/10 → 13/21-d)
  2. Multi-resolution HexPlane grids (6 planes × L levels, feat_dim C)
     with bilinear interpolation and product fusion across complementary pairs
  3. Concatenation: [grid_feats, xyz_PE, t_PE]
  4. MLP decoder → (d_xyz: 3, d_rotation: 4, d_scaling: 3)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Positional Encoding (Fourier features)
# ──────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Fourier positional encoding identical to NeRF / DeformNetwork.

    For input dimension *d* and *L* frequency bands:
        output = [x, sin(2^0 π x), cos(2^0 π x), ..., sin(2^{L-1} π x), cos(2^{L-1} π x)]
        output_dim = d + 2 * d * L  (when include_input=True)

    Parameters
    ----------
    input_dims : int
        Dimensionality of raw input (e.g. 3 for xyz, 1 for t).
    num_freqs : int
        Number of frequency bands *L*.
    include_input : bool
        Whether to prepend the raw input to the encoding.
    log_sampling : bool
        If True, frequencies are 2^{0..L-1}; otherwise linearly spaced.
    """

    def __init__(
        self,
        input_dims: int = 3,
        num_freqs: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), steps=num_freqs)
        # Register as buffer so it moves to GPU with the module
        self.register_buffer("freq_bands", freq_bands)  # (L,)

        self._out_dim = (1 + 2 * num_freqs) * input_dims if include_input else 2 * num_freqs * input_dims

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(N, input_dims)``

        Returns
        -------
        ``(N, out_dim)``
        """
        # x_scaled: (N, input_dims, L)
        x_scaled = x.unsqueeze(-1) * self.freq_bands  # broadcast
        # sin/cos: each (N, input_dims, L) → reshape to (N, input_dims * L)
        sin_part = torch.sin(x_scaled).reshape(x.shape[0], -1)
        cos_part = torch.cos(x_scaled).reshape(x.shape[0], -1)
        parts = [sin_part, cos_part]
        if self.include_input:
            parts = [x] + parts
        return torch.cat(parts, dim=-1)  # (N, out_dim)


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

# Indices into _PLANE_PAIRS whose W-axis (dim=-1) is the temporal axis.
# XY(0), XZ(1), YZ(3) are pure-spatial; XT(2), YT(4), ZT(5) are space-time.
_TEMPORAL_PLANE_INDICES: frozenset = frozenset({2, 4, 5})


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
        spatial_resolutions: Sequence[int] = (64, 128, 256),
        time_resolutions: Sequence[int] = (64, 128, 256),
        feat_dim: int = 16,
        fusion: str = "concat",
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
            # 3 complementary pairs → C each, *concatenated* (not summed) per level
            self._out_dim = self.num_levels * 3 * feat_dim

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
                # pair 0: XY(0) ⊙ ZT(5)
                # pair 1: XZ(1) ⊙ YT(4)
                # pair 2: XT(2) ⊙ YZ(3)
                # Concatenate (NOT sum) the 3 products → (N, 3C)
                p0 = plane_feats[0] * plane_feats[5]  # XY ⊙ ZT  (N, C)
                p1 = plane_feats[1] * plane_feats[4]  # XZ ⊙ YT  (N, C)
                p2 = plane_feats[2] * plane_feats[3]  # XT ⊙ YZ  (N, C)
                level_feats.append(torch.cat([p0, p1, p2], dim=-1))  # (N, 3C)

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

    def compute_plane_tv_split(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """TV loss split by axis type: (spatial_tv, temporal_tv).

        For pure-spatial planes (XY, XZ, YZ) both H and W differencing
        goes into *spatial_tv*.  For space-time planes (XT, YT, ZT) the
        H-direction (spatial axis) goes into *spatial_tv* and the
        W-direction (temporal axis) goes into *temporal_tv*.

        Returns
        -------
        spatial_tv : torch.Tensor  — scalar
        temporal_tv : torch.Tensor — scalar
        """
        dev = next(self.parameters()).device
        spatial_tv = torch.tensor(0.0, device=dev)
        temporal_tv = torch.tensor(0.0, device=dev)
        for lvl_planes in self.planes:
            for pidx, p in enumerate(lvl_planes):
                # p: (1, C, H, W)
                h_tv = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
                w_tv = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
                if pidx in _TEMPORAL_PLANE_INDICES:
                    # H = spatial axis, W = temporal axis
                    spatial_tv = spatial_tv + h_tv
                    temporal_tv = temporal_tv + w_tv
                else:
                    # Both axes are spatial
                    spatial_tv = spatial_tv + h_tv + w_tv
        return spatial_tv, temporal_tv


# ──────────────────────────────────────────────────────────────────────
#  MLP Decoder
# ──────────────────────────────────────────────────────────────────────

class HexPlaneMLPDecoder(nn.Module):
    """MLP that maps [grid_feats, xyz_PE, t_PE] to deformation outputs.

    Output layout: ``(d_xyz: 3, d_rotation: 4, d_scaling: 3)`` = 10 dims.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
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
#  Composite Network: HexPlane + PE + MLP Decoder
# ──────────────────────────────────────────────────────────────────────

class HexPlaneDeformNetwork(nn.Module):
    """4DGS-style deformation network: PE + multi-res HexPlane → MLP decoder.

    Accepts raw ``(xyz, t)`` inputs, computes positional encodings for both
    spatial and temporal coordinates, queries the HexPlane feature field,
    and concatenates ``[grid_feats, xyz_PE, t_PE]`` before the MLP decoder.

    This matches the architecture in the 4DGS paper (Wu et al., CVPR 2024).

    Parameters
    ----------
    spatial_resolutions : list[int]
        Multi-resolution grid sizes for spatial axes.
    time_resolutions : list[int]
        Multi-resolution grid sizes for the temporal axis.
    feat_dim : int
        Feature channels per plane per resolution level.
    xyz_multires : int
        Number of Fourier frequency bands for spatial PE (default 10, → 63-d).
    t_multires : int
        Number of Fourier frequency bands for temporal PE (default 6, → 13-d).
        Use 6 for blender/D-NeRF scenes, 10 for real-world scenes.
    mlp_hidden_dim : int
        Hidden layer width for the MLP decoder.
    mlp_num_hidden : int
        Number of hidden layers in the MLP decoder.
    fusion : str
        HexPlane fusion strategy: ``"concat"`` or ``"product"``.
    init_scale : float
        Plane parameter initialisation scale.
    is_blender : bool
        If True, use t_multires=6 (matching DeformNetwork behaviour).
    is_6dof : bool
        Placeholder for SE(3) output mode (not yet implemented).
    """

    def __init__(
        self,
        spatial_resolutions: Sequence[int] = (64, 128, 256),
        time_resolutions: Sequence[int] = (64, 128, 256),
        feat_dim: int = 16,
        xyz_multires: int = 10,
        t_multires: int = 10,
        mlp_hidden_dim: int = 128,
        mlp_num_hidden: int = 2,
        fusion: str = "concat",
        init_scale: float = 0.1,
        is_blender: bool = False,
        is_6dof: bool = False,
    ) -> None:
        super().__init__()
        self.is_6dof = is_6dof
        self.is_blender = is_blender

        # ── Positional encodings ──────────────────────────────────────
        # Match original DeformNetwork: blender uses t_multires=6
        _t_multires = 6 if is_blender else t_multires
        self.pe_xyz = PositionalEncoding(input_dims=3, num_freqs=xyz_multires)
        self.pe_t = PositionalEncoding(input_dims=1, num_freqs=_t_multires)

        # ── HexPlane grids ────────────────────────────────────────────
        self.hexplane = HexPlaneField(
            spatial_resolutions=spatial_resolutions,
            time_resolutions=time_resolutions,
            feat_dim=feat_dim,
            fusion=fusion,
            init_scale=init_scale,
        )

        # ── MLP decoder ──────────────────────────────────────────────
        # Input = grid_feats + xyz_PE + t_PE
        decoder_in_dim = self.hexplane.out_dim + self.pe_xyz.out_dim + self.pe_t.out_dim
        out_dim = 10  # (d_xyz: 3, d_rotation: 4, d_scaling: 3)
        self.decoder = HexPlaneMLPDecoder(
            in_dim=decoder_in_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden,
            out_dim=out_dim,
        )

        # ── AABB for normalising xyz to [-1, 1]. Updated via set_aabb().
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
        and mapped to [-1, 1].

        Out-of-range spatial coordinates are wrapped back into [-1, 1) using
        modulo arithmetic rather than being clamped at the border.  This
        prevents ``grid_sample`` border-clamping artefacts when a Gaussian
        temporarily leaves its cluster AABB due to deformation:

        .. code-block:: text

            norm ∈ [-1, 3]  →  wrap = ((norm + 1) % 2) - 1  ∈ [-1, 1)
        """
        xyz_norm = 2.0 * (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8) - 1.0
        # Cyclic wrap: maps any value back into [-1, 1) with period 2.
        xyz_norm = ((xyz_norm + 1.0) % 2.0) - 1.0
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
        # 1. Normalise → grid query
        xyzt = self._normalise_xyzt(xyz, t)         # (N, 4)
        xyz_norm = xyzt[:, :3]                       # cluster-local coords in [-1, 1)
        t_norm   = xyzt[:, 3:4]                      # time in [-1, 1]
        grid_feats = self.hexplane(xyzt)             # (N, grid_dim)

        # 2. Positional encoding of *normalised* inputs so that PE frequencies
        #    are consistent across all students regardless of cluster AABB size.
        #    Using raw world coords here would mis-align PE scales between students
        #    that cover different sub-regions of the scene, negating the resolution
        #    gain from per-cluster AABB normalisation.
        xyz_pe = self.pe_xyz(xyz_norm)               # (N, 63) for multires=10
        t_pe   = self.pe_t(t_norm)                   # (N, 13) for multires=6

        # 3. Concatenate and decode
        decoder_input = torch.cat([grid_feats, xyz_pe, t_pe], dim=-1)
        out = self.decoder(decoder_input)            # (N, 10)

        d_xyz = out[:, :3]
        d_rotation = out[:, 3:7]
        d_scaling = out[:, 7:10]
        return d_xyz, d_rotation, d_scaling

    # ---- Regularisation ---------------------------------------------------

    def get_plane_tv_loss(self) -> torch.Tensor:
        return self.hexplane.compute_plane_tv()

    def get_plane_l1_loss(self) -> torch.Tensor:
        return self.hexplane.compute_plane_l1()

    def get_plane_tv_loss_split(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (spatial_tv, temporal_tv) from underlying HexPlaneField."""
        return self.hexplane.compute_plane_tv_split()


# ──────────────────────────────────────────────────────────────────────
#  Path A: BatchedHexPlaneField
#  Stores K students' planes as (K, C, H, W) parameters.
#  grid_sample call count: num_levels * 6  (constant w.r.t. K)
# ──────────────────────────────────────────────────────────────────────

class BatchedHexPlaneField(nn.Module):
    """Multi-resolution HexPlane grids for K uniform-architecture students.

    Replaces ``K`` independent ``HexPlaneField`` instances with a single
    module whose plane parameters have shape ``(K, C, H, W)`` instead of
    ``(1, C, H, W)``.  A single batched ``F.grid_sample`` call handles all
    K students simultaneously, reducing the total kernel count from
    ``K * num_levels * 6`` to ``num_levels * 6`` (constant w.r.t. K).

    Parameters
    ----------
    K : int
        Number of student networks (batch dimension).
    spatial_resolutions, time_resolutions, feat_dim, fusion, init_scale :
        Same semantics as :class:`HexPlaneField`.
    """

    FUSION_MODES = ("concat", "product")

    def __init__(
        self,
        K: int,
        spatial_resolutions: Sequence[int] = (64, 128, 192),
        time_resolutions: Sequence[int] = (64, 128, 192),
        feat_dim: int = 12,
        fusion: str = "concat",
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(spatial_resolutions) == len(time_resolutions)
        assert fusion in self.FUSION_MODES

        self.K = K
        self.spatial_resolutions = list(spatial_resolutions)
        self.time_resolutions = list(time_resolutions)
        self.num_levels = len(spatial_resolutions)
        self.feat_dim = feat_dim
        self.fusion = fusion

        # self.planes[level][plane_idx] : nn.Parameter of shape (K, C, H, W)
        self.planes = nn.ModuleList()
        for lvl in range(self.num_levels):
            s_res = spatial_resolutions[lvl]
            t_res = time_resolutions[lvl]
            axis_res = [s_res, s_res, s_res, t_res]
            level_planes = nn.ParameterList()
            for di, dj in _PLANE_PAIRS:
                h, w = axis_res[di], axis_res[dj]
                param = nn.Parameter(init_scale * torch.randn(K, feat_dim, h, w))
                level_planes.append(param)
            self.planes.append(level_planes)

        if fusion == "concat":
            self._out_dim = self.num_levels * 6 * feat_dim
        else:
            self._out_dim = self.num_levels * 3 * feat_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, xyzt_batched: torch.Tensor) -> torch.Tensor:
        """Sample and fuse plane features for all K students in one pass.

        Parameters
        ----------
        xyzt_batched : ``(K, N_max, 4)``
            Per-cluster normalised coordinates in ``[-1, 1]``.

        Returns
        -------
        ``(K, N_max, out_dim)``
        """
        K, N_max, _ = xyzt_batched.shape
        level_feats: List[torch.Tensor] = []

        for lvl in range(self.num_levels):
            plane_feats: List[torch.Tensor] = []
            for pidx, (di, dj) in enumerate(_PLANE_PAIRS):
                plane = self.planes[lvl][pidx]  # (K, C, H, W)
                # grid_sample grid shape: (N_batch, H_out, W_out, 2)
                # Here N_batch=K, H_out=N_max, W_out=1
                coords = torch.stack(
                    [xyzt_batched[..., dj], xyzt_batched[..., di]], dim=-1
                )  # (K, N_max, 2)
                grid = coords.unsqueeze(2)  # (K, N_max, 1, 2)
                feat = F.grid_sample(
                    plane, grid,
                    align_corners=True, mode="bilinear", padding_mode="border",
                )  # (K, C, N_max, 1)
                plane_feats.append(feat.squeeze(-1).permute(0, 2, 1))  # (K, N_max, C)

            if self.fusion == "concat":
                level_feats.append(torch.cat(plane_feats, dim=-1))  # (K, N_max, 6C)
            else:
                p0 = plane_feats[0] * plane_feats[5]
                p1 = plane_feats[1] * plane_feats[4]
                p2 = plane_feats[2] * plane_feats[3]
                level_feats.append(torch.cat([p0, p1, p2], dim=-1))  # (K, N_max, 3C)

        return torch.cat(level_feats, dim=-1)  # (K, N_max, out_dim)

    # ── Regularisation helpers ────────────────────────────────────────
    # O(1) kernel calls w.r.t. K — diff ops broadcast over the K dimension.

    def compute_plane_tv(self) -> torch.Tensor:
        """TV loss across all K students' planes — single batched computation."""
        tv = torch.tensor(0.0, device=next(self.parameters()).device)
        for lvl_planes in self.planes:
            for p in lvl_planes:  # (K, C, H, W)
                tv = tv + (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
                tv = tv + (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
        return tv

    def compute_plane_l1(self) -> torch.Tensor:
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for lvl_planes in self.planes:
            for p in lvl_planes:
                l1 = l1 + p.abs().mean()
        return l1

    def compute_plane_tv_split(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (spatial_tv, temporal_tv)."""
        dev = next(self.parameters()).device
        spatial_tv = torch.tensor(0.0, device=dev)
        temporal_tv = torch.tensor(0.0, device=dev)
        for lvl_planes in self.planes:
            for pidx, p in enumerate(lvl_planes):
                h_tv = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
                w_tv = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
                if pidx in _TEMPORAL_PLANE_INDICES:
                    spatial_tv = spatial_tv + h_tv
                    temporal_tv = temporal_tv + w_tv
                else:
                    spatial_tv = spatial_tv + h_tv + w_tv
        return spatial_tv, temporal_tv


# ──────────────────────────────────────────────────────────────────────
#  Path B: BatchedHexPlaneMLPDecoder
#  Weight matrices stored as (K, out_features, in_features).
#  Forward uses torch.bmm → cuBLAS batched GEMM (2 kernel calls total).
# ──────────────────────────────────────────────────────────────────────

class BatchedHexPlaneMLPDecoder(nn.Module):
    """Batched MLP decoder for K uniform-architecture students.

    Weight matrices are stored as ``(K, out_features, in_features)`` tensors
    matching the ``nn.Linear`` convention.  The forward pass uses
    ``torch.bmm(x, W.T) + b`` so a single cuBLAS batched-GEMM kernel handles
    all K students simultaneously.

    Parameters
    ----------
    K : int
        Number of students (batch dimension).
    in_dim, hidden_dim, num_hidden_layers, out_dim :
        Same semantics as :class:`HexPlaneMLPDecoder`.
    """

    def __init__(
        self,
        K: int,
        in_dim: int,
        hidden_dim: int = 146,
        num_hidden_layers: int = 2,
        out_dim: int = 10,
    ) -> None:
        super().__init__()
        self.K = K
        self.num_hidden_layers = num_hidden_layers

        # Layer 0: in_dim → hidden_dim
        # Shape follows nn.Linear convention: (out_features, in_features)
        self.W0 = nn.Parameter(torch.empty(K, hidden_dim, in_dim))
        self.b0 = nn.Parameter(torch.zeros(K, hidden_dim))

        # Hidden-to-hidden layers: hidden_dim → hidden_dim
        self.W_mid = nn.ParameterList([
            nn.Parameter(torch.empty(K, hidden_dim, hidden_dim))
            for _ in range(num_hidden_layers - 1)
        ])
        self.b_mid = nn.ParameterList([
            nn.Parameter(torch.zeros(K, hidden_dim))
            for _ in range(num_hidden_layers - 1)
        ])

        # Output layer: zero-init (same as HexPlaneMLPDecoder)
        self.W_out = nn.Parameter(torch.zeros(K, out_dim, hidden_dim))
        self.b_out = nn.Parameter(torch.zeros(K, out_dim))

        # Kaiming uniform init for input and hidden layers
        # bound = 1/sqrt(fan_in) is derived from kaiming_uniform_(a=sqrt(5))
        self._init_weight(self.W0, fan_in=in_dim)
        for w in self.W_mid:
            self._init_weight(w, fan_in=hidden_dim)

    @staticmethod
    def _init_weight(w: nn.Parameter, fan_in: int) -> None:
        """Kaiming uniform init matching nn.Linear default (a=sqrt(5))."""
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(w, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(K, N_max, in_dim)``

        Returns
        -------
        ``(K, N_max, out_dim)``
        """
        # bmm: (K, N, in) @ (K, in, hid) = (K, N, hid)
        h = torch.bmm(x, self.W0.transpose(-2, -1)) + self.b0.unsqueeze(1)
        h = F.relu(h)

        for w_mid, b_mid in zip(self.W_mid, self.b_mid):
            h = torch.bmm(h, w_mid.transpose(-2, -1)) + b_mid.unsqueeze(1)
            h = F.relu(h)

        out = torch.bmm(h, self.W_out.transpose(-2, -1)) + self.b_out.unsqueeze(1)
        return out  # (K, N_max, 10)


# ──────────────────────────────────────────────────────────────────────
#  BatchedHexPlaneDeformNetwork
#  Combines Path A + Path B with vectorised per-cluster AABB normalisation.
# ──────────────────────────────────────────────────────────────────────

class BatchedHexPlaneDeformNetwork(nn.Module):
    """4DGS-style deformation network for K uniform-architecture students.

    Designed for scaling experiments where all students share the same
    architecture (e.g. all High Tier).  Key properties:

    * **Path A** — :class:`BatchedHexPlaneField`: ``K × 18`` grid_sample calls
      reduced to ``18`` (constant w.r.t. K).
    * **Path B** — :class:`BatchedHexPlaneMLPDecoder`: ``K`` independent
      ``nn.Linear`` calls replaced by 2 batched ``torch.bmm`` calls.
    * **Vectorised AABB normalisation**: per-cluster AABB stored as ``(K, 3)``
      buffers; normalisation computed in one broadcast operation.
    * **No ParallelBackwardHandle needed**: batched autograd naturally
      propagates gradients through batched ops.

    Parameters
    ----------
    K : int
        Number of student networks.
    spatial_resolutions, time_resolutions, feat_dim, mlp_hidden_dim,
    mlp_num_hidden, fusion, is_blender, is_6dof, init_scale :
        Same semantics as :class:`HexPlaneDeformNetwork`.
    """

    def __init__(
        self,
        K: int,
        spatial_resolutions: Sequence[int] = (64, 128, 192),
        time_resolutions: Sequence[int] = (64, 128, 192),
        feat_dim: int = 12,
        mlp_hidden_dim: int = 146,
        mlp_num_hidden: int = 2,
        fusion: str = "concat",
        is_blender: bool = False,
        is_6dof: bool = False,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = K
        self.is_blender = is_blender
        self.is_6dof = is_6dof

        # Shared PE (deterministic, not learned — same for all students)
        _t_multires = 6 if is_blender else 10
        self.pe_xyz = PositionalEncoding(input_dims=3, num_freqs=10)
        self.pe_t = PositionalEncoding(input_dims=1, num_freqs=_t_multires)

        # Path A: batched HexPlane grids
        self.hexplane = BatchedHexPlaneField(
            K=K,
            spatial_resolutions=spatial_resolutions,
            time_resolutions=time_resolutions,
            feat_dim=feat_dim,
            fusion=fusion,
            init_scale=init_scale,
        )

        # Path B: batched MLP decoder
        decoder_in_dim = (
            self.hexplane.out_dim + self.pe_xyz.out_dim + self.pe_t.out_dim
        )
        self.decoder = BatchedHexPlaneMLPDecoder(
            K=K,
            in_dim=decoder_in_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden,
            out_dim=10,
        )

        # Per-student AABB buffers: (K, 3)
        self.register_buffer(
            "aabb_mins",
            torch.stack([torch.tensor([-1.0, -1.0, -1.0])] * K),
        )
        self.register_buffer(
            "aabb_maxs",
            torch.stack([torch.tensor([1.0, 1.0, 1.0])] * K),
        )

    # ── AABB management ──────────────────────────────────────────────

    def set_aabb_single(
        self, k: int, points: torch.Tensor, padding: float = 0.1
    ) -> None:
        """Set AABB for student *k* from a point cloud."""
        pmin = points.min(dim=0).values
        pmax = points.max(dim=0).values
        extent = (pmax - pmin).clamp(min=1e-6)
        self.aabb_mins[k] = pmin - padding * extent
        self.aabb_maxs[k] = pmax + padding * extent

    def set_aabb_all(
        self, points: torch.Tensor, padding: float = 0.1
    ) -> None:
        """Set the same global AABB for all K students."""
        pmin = points.min(dim=0).values
        pmax = points.max(dim=0).values
        extent = (pmax - pmin).clamp(min=1e-6)
        self.aabb_mins[:] = (pmin - padding * extent).unsqueeze(0)
        self.aabb_maxs[:] = (pmax + padding * extent).unsqueeze(0)

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        cluster_xyz: List[torch.Tensor],
        cluster_t: List[torch.Tensor],
        cluster_counts: List[int],
    ) -> torch.Tensor:
        """Batched deformation forward for all K students.

        Parameters
        ----------
        cluster_xyz : list of K tensors, each ``(N_k, 3)``
        cluster_t   : list of K tensors, each ``(N_k, 1)``
        cluster_counts : list of K ints (N_k per cluster)

        Returns
        -------
        out : ``(K, N_max, 10)``
            Packed output. Slice ``out[k, :N_k, :]`` to get student k's result.
            Column layout: ``[:3]`` d_xyz, ``[3:7]`` d_rotation, ``[7:10]`` d_scaling.
        """
        K = self.K
        device = self.aabb_mins.device
        N_max = max(cluster_counts) if max(cluster_counts) > 0 else 1
        # Infer dtype from first non-empty cluster
        dtype = next(
            (xyz.dtype for xyz, n in zip(cluster_xyz, cluster_counts) if n > 0),
            torch.float32,
        )

        # 1. Pad inputs to (K, N_max, dim)
        xyz_padded = torch.zeros(K, N_max, 3, device=device, dtype=dtype)
        t_padded = torch.zeros(K, N_max, 1, device=device, dtype=dtype)
        for k in range(K):
            n = cluster_counts[k]
            if n > 0:
                xyz_padded[k, :n] = cluster_xyz[k]
                t_padded[k, :n] = cluster_t[k]

        # 2. Vectorised per-cluster AABB normalisation
        # aabb_mins/maxs: (K, 3) → broadcast over N_max via unsqueeze(1)
        extent = (self.aabb_maxs - self.aabb_mins).clamp(min=1e-8).unsqueeze(1)
        xyz_norm = (
            2.0 * (xyz_padded - self.aabb_mins.unsqueeze(1)) / extent - 1.0
        )  # (K, N_max, 3)
        # Cyclic wrap: maps any value back into [-1, 1) with period 2
        xyz_norm = ((xyz_norm + 1.0) % 2.0) - 1.0
        t_norm = 2.0 * t_padded - 1.0  # (K, N_max, 1)

        # 3. Assemble xyzt: (K, N_max, 4)
        xyzt_norm = torch.cat([xyz_norm, t_norm], dim=-1)

        # 4. Path A: batched HexPlane — 18 grid_sample calls, constant w.r.t. K
        grid_feats = self.hexplane(xyzt_norm)  # (K, N_max, grid_out_dim)

        # 5. Shared PE on normalised coords — flatten then unflatten
        xyz_pe = self.pe_xyz(
            xyz_norm.reshape(K * N_max, 3)
        ).reshape(K, N_max, -1)  # (K, N_max, 63)
        t_pe = self.pe_t(
            t_norm.reshape(K * N_max, 1)
        ).reshape(K, N_max, -1)  # (K, N_max, 21)

        # 6. Path B: concatenate → batched MLP (2 bmm calls total)
        decoder_input = torch.cat([grid_feats, xyz_pe, t_pe], dim=-1)
        out = self.decoder(decoder_input)  # (K, N_max, 10)

        return out

    # ── Regularisation ───────────────────────────────────────────────

    def get_plane_tv_loss(self) -> torch.Tensor:
        return self.hexplane.compute_plane_tv()

    def get_plane_l1_loss(self) -> torch.Tensor:
        return self.hexplane.compute_plane_l1()

    def get_plane_tv_loss_split(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hexplane.compute_plane_tv_split()
