import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
from utils.hexplane_utils import HexPlaneDeformNetwork
import os
from typing import Optional, Sequence, Tuple
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
            #  'lr': training_args.position_lr_init * self.spatial_lr_scale,
            'lr': 0.2, #0.0008
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


class DeformModel_4DGS:
    """4DGS-style deformation model: PE + multi-resolution HexPlane + MLP decoder.

    Drop-in replacement for :class:`DeformModel` with the same external interface
    (``step``, ``train_setting``, ``save_weights``, ``load_weights``,
    ``update_learning_rate``).

    Architecture matches the 4DGS paper (Wu et al., CVPR 2024):
      xyz PE (multires=10, 63-d) + t PE (multires=6/10, 13/21-d)
      + 3-level HexPlane grids (feat_dim=16, product fusion → 48-d)
      → MLP decoder (128-hidden, 2 layers) → (d_xyz, d_rot, d_scale)

    Extra capabilities:
    * ``set_aabb(points)`` — compute normalisation AABB from a point cloud.
    * ``get_tv_loss()`` / ``get_l1_loss()`` — HexPlane regularisation terms.

    Parameters
    ----------
    is_blender : bool
        If True, use t_multires=6 (D-NeRF); otherwise t_multires=10.
    spatial_resolutions : tuple[int, ...]
        Multi-res spatial grid sizes (default ``(64, 128, 256)``).
    time_resolutions : tuple[int, ...]
        Multi-res temporal grid sizes (default ``(64, 128, 256)``).
    feat_dim : int
        Feature channels per plane per resolution level (default 16).
    mlp_hidden_dim : int
        Hidden width of the MLP decoder (default 128).
    mlp_num_hidden : int
        Number of hidden layers in the MLP decoder (default 2).
    fusion : str
        HexPlane fusion mode: ``"product"`` (default) or ``"concat"``.
    is_6dof : bool
        Placeholder for SE(3) output mode.
    """

    def __init__(
        self,
        is_blender: bool = False,
        is_6dof: bool = False,
        spatial_resolutions: Sequence[int] = (64, 128, 256),
        time_resolutions: Sequence[int] = (64, 128, 256),
        feat_dim: int = 16,
        mlp_hidden_dim: int = 128,
        mlp_num_hidden: int = 2,
        fusion: str = "concat",
    ) -> None:
        self.deform = HexPlaneDeformNetwork(
            spatial_resolutions=spatial_resolutions,
            time_resolutions=time_resolutions,
            feat_dim=feat_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_hidden=mlp_num_hidden,
            fusion=fusion,
            is_blender=is_blender,
            is_6dof=is_6dof,
        ).cuda()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.spatial_lr_scale = 5

    # ── AABB ──────────────────────────────────────────────────────────

    def set_aabb(self, points: torch.Tensor, padding: float = 0.1) -> None:
        """Propagate AABB computation to the underlying network."""
        self.deform.set_aabb(points, padding=padding)

    # ── Forward ───────────────────────────────────────────────────────

    def step(
        self, xyz: torch.Tensor, time_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.deform(xyz, time_emb)

    # ── Training configuration ────────────────────────────────────────

    def train_setting(self, training_args) -> None:
        # Separate parameter groups: planes get a higher initial lr than MLP.
        # Grid-based methods (K-Planes, 4DGS, Instant-NGP) require high lr
        # for planes because each parameter only receives gradients from
        # nearby sample points.
        plane_params = list(self.deform.hexplane.parameters())
        mlp_params = list(self.deform.decoder.parameters())
        # Also include PE buffers that may become parameters in future
        pe_params = (
            list(self.deform.pe_xyz.parameters())
            + list(self.deform.pe_t.parameters())
        )
        # Read LR params — fall back to sensible defaults if not present
        _plane_lr_init = getattr(training_args, "hex_plane_lr_init", 0.02)
        _plane_lr_final = getattr(training_args, "hex_plane_lr_final", 0.002)
        _mlp_lr_init = getattr(training_args, "hex_mlp_lr_init", 0.001)
        _mlp_lr_final = getattr(training_args, "hex_mlp_lr_final", 0.00001)

        l = [
            {"params": plane_params, "lr": _plane_lr_init, "name": "deform_planes"},
            {"params": mlp_params + pe_params, "lr": _mlp_lr_init, "name": "deform_mlp"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Independent LR schedulers for planes and MLP.
        _max_steps = training_args.deform_lr_max_steps
        self._plane_lr_func = get_expon_lr_func(
            lr_init=_plane_lr_init,
            lr_final=_plane_lr_final,
            lr_delay_mult=0.01,
            max_steps=_max_steps,
        )
        self._mlp_lr_func = get_expon_lr_func(
            lr_init=_mlp_lr_init,
            lr_final=_mlp_lr_final,
            lr_delay_mult=0.01,
            max_steps=_max_steps,
        )

    # ── Persistence ───────────────────────────────────────────────────

    def save_weights(self, model_path: str, iteration: int) -> None:
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, "deform.pth"))

    def load_weights(self, model_path: str, iteration: int = -1) -> None:
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "deform/iteration_{}/deform.pth".format(loaded_iter)
        )
        self.deform.load_state_dict(torch.load(weights_path))

    # ── LR scheduling ─────────────────────────────────────────────────

    def update_learning_rate(self, iteration: int) -> Optional[float]:
        plane_lr = self._plane_lr_func(iteration)
        mlp_lr = self._mlp_lr_func(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_planes":
                param_group["lr"] = plane_lr
            elif param_group["name"] == "deform_mlp":
                param_group["lr"] = mlp_lr
        return plane_lr

    # ── Regularisation losses ─────────────────────────────────────────

    def get_tv_loss(self) -> torch.Tensor:
        """Total Variation loss on HexPlane grids."""
        return self.deform.get_plane_tv_loss()

    def get_l1_loss(self) -> torch.Tensor:
        """L1 sparsity loss on HexPlane grid parameters."""
        return self.deform.get_plane_l1_loss()
