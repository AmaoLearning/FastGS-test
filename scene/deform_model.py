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

    def get_regularization_loss(self) -> torch.Tensor:
        """Get TV and L1 regularization"""
        return 1e-3 * self.deform.get_plane_tv_loss() + 1e-4 * self.deform.get_plane_l1_loss()


class ClusteredDeformModel:
    """Multi-deform model for clustered dynamic Gaussians.
    
    This class manages multiple deformation fields, each responsible for a cluster
    of dynamic Gaussians. Implements teacher-student distillation from a single
    teacher model to multiple student models.
    
    Architecture:
    - 1 teacher deform field (original 4DGS model, frozen after clustering)
    - K student deform fields (one per cluster, configurable capacity)
    - Student model architecture is manually configurable
    
    Training objectives:
    1. Visual quality loss (L1 + SSIM) - same as original
    2. Knowledge distillation loss - student predictions match teacher
    """
    
    def __init__(
        self,
        n_clusters: int,
        is_blender: bool = False,
        is_6dof: bool = False,
        # Teacher config (original 4DGS)
        teacher_spatial_resolutions: Sequence[int] = (64, 128, 256),
        teacher_time_resolutions: Sequence[int] = (64, 128, 256),
        teacher_feat_dim: int = 16,
        teacher_mlp_hidden_dim: int = 128,
        teacher_mlp_num_hidden: int = 2,
        # Student config (manually configured)
        student_feat_dim: int = 8,
        student_spatial_resolutions: Sequence[int] = (64, 128),
        student_time_resolutions: Sequence[int] = (64, 128),
        student_mlp_hidden_dim: int = 64,
        student_mlp_num_hidden: int = 2,
        fusion: str = "concat",
    ) -> None:
        self.n_clusters = n_clusters
        self.is_blender = is_blender
        self.is_6dof = is_6dof
        self.fusion = fusion
        
        # Teacher model (original capacity)
        self.teacher = HexPlaneDeformNetwork(
            spatial_resolutions=teacher_spatial_resolutions,
            time_resolutions=teacher_time_resolutions,
            feat_dim=teacher_feat_dim,
            mlp_hidden_dim=teacher_mlp_hidden_dim,
            mlp_num_hidden=teacher_mlp_num_hidden,
            fusion=fusion,
            is_blender=is_blender,
            is_6dof=is_6dof,
        ).cuda()
        self.teacher.eval()  # Teacher is frozen after clustering
        
        # Student models (configurable architecture)
        self.students = nn.ModuleList()
        for _ in range(n_clusters):
            student = HexPlaneDeformNetwork(
                spatial_resolutions=student_spatial_resolutions,
                time_resolutions=student_time_resolutions,
                feat_dim=student_feat_dim,
                mlp_hidden_dim=student_mlp_hidden_dim,
                mlp_num_hidden=student_mlp_num_hidden,
                fusion=fusion,
                is_blender=is_blender,
                is_6dof=is_6dof,
            ).cuda()
            self.students.append(student)
        
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.spatial_lr_scale = 5
        
        # Cluster assignments (updated during training)
        self._cluster_labels: Optional[torch.Tensor] = None  # (N,) int32
        
    def set_aabb(self, points: torch.Tensor, padding: float = 0.1) -> None:
        """Set AABB for all deform models."""
        self.teacher.set_aabb(points, padding=padding)
        for student in self.students:
            student.set_aabb(points, padding=padding)
    
    def set_cluster_labels(self, cluster_labels: torch.Tensor) -> None:
        """Set cluster labels for Gaussian assignment."""
        self._cluster_labels = cluster_labels  # (N,) int32, -1 for static
    
    def step(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for clustered deformation (parallel inference).
        
        Uses batched parallel inference: all student models receive their
        respective cluster inputs in a single concatenated batch, enabling
        CUDA kernel fusion and overlapping memory operations.
        
        Args:
            xyz: Gaussian positions (N, 3)
            time_emb: Time embeddings (N, 1) or (1, 1)
            cluster_ids: Cluster assignments for each Gaussian (N,) or None to use self._cluster_labels
        
        Returns:
            d_xyz, d_rotation, d_scaling: Deformation outputs (N, 3), (N, 4), (N, 3)
        """
        if cluster_ids is None:
            cluster_ids = self._cluster_labels
        
        if cluster_ids is None:
            raise ValueError("No cluster labels provided. Set cluster_labels first.")
        
        N = xyz.shape[0]
        device = xyz.device
        dtype = xyz.dtype
        
        # Pre-allocate output tensors
        d_xyz = torch.zeros(N, 3, device=device, dtype=dtype)
        d_rotation = torch.zeros(N, 4, device=device, dtype=dtype)
        d_scaling = torch.zeros(N, 3, device=device, dtype=dtype)
        
        # Determine if time_emb needs per-point indexing
        use_per_point_time = time_emb.shape[0] == N
        
        # Gather inputs for all clusters in a single pass
        # Store as list of tuples for parallel application
        cluster_inputs = []
        cluster_masks = []
        
        for cluster_id in range(self.n_clusters):
            mask = (cluster_ids == cluster_id)
            count = mask.sum().item()
            if count == 0:
                continue
            
            cluster_xyz = xyz[mask]
            cluster_time = time_emb[mask] if use_per_point_time else time_emb.expand(count, -1)
            
            cluster_inputs.append((cluster_xyz, cluster_time))
            cluster_masks.append(mask)
        
        # Parallel forward pass using list comprehension
        # All inputs are on same device (CUDA), enabling kernel overlap
        if len(cluster_inputs) == 0:
            return d_xyz, d_rotation, d_scaling
        
        # Batch inference: process all clusters
        # Each student model processes its cluster independently
        cluster_outputs = [
            student(xyz_c, time_c)
            for student, (xyz_c, time_c) in zip(self.students, cluster_inputs)
        ]
        
        # Scatter results back to output tensors
        for i, (d_xyz_c, d_rotation_c, d_scaling_c) in enumerate(cluster_outputs):
            mask = cluster_masks[i]
            d_xyz[mask] = d_xyz_c
            d_rotation[mask] = d_rotation_c
            d_scaling[mask] = d_scaling_c
        
        return d_xyz, d_rotation, d_scaling
    
    def step_teacher(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through teacher model (for distillation target)."""
        return self.teacher(xyz, time_emb)
    
    def train_setting(self, training_args) -> None:
        """Initialize optimizers for all student models."""
        param_groups = []
        
        for cluster_id, student in enumerate(self.students):
            plane_params = list(student.hexplane.parameters())
            mlp_params = list(student.decoder.parameters())
            pe_params = (
                list(student.pe_xyz.parameters())
                + list(student.pe_t.parameters())
            )
            
            _plane_lr_init = getattr(training_args, "hex_plane_lr_init", 0.02)
            _plane_lr_final = getattr(training_args, "hex_plane_lr_final", 0.002)
            _mlp_lr_init = getattr(training_args, "hex_mlp_lr_init", 0.001)
            _mlp_lr_final = getattr(training_args, "hex_mlp_lr_final", 0.00001)
            
            param_groups.extend([
                {"params": plane_params, "lr": _plane_lr_init, "name": f"student_{cluster_id}_planes"},
                {"params": mlp_params + pe_params, "lr": _mlp_lr_init, "name": f"student_{cluster_id}_mlp"},
            ])
        
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
        # LR schedulers
        _max_steps = training_args.deform_lr_max_steps
        self._plane_lr_func = get_expon_lr_func(
            lr_init=getattr(training_args, "hex_plane_lr_init", 0.02),
            lr_final=getattr(training_args, "hex_plane_lr_final", 0.002),
            lr_delay_mult=0.01,
            max_steps=_max_steps,
        )
        self._mlp_lr_func = get_expon_lr_func(
            lr_init=getattr(training_args, "hex_mlp_lr_init", 0.001),
            lr_final=getattr(training_args, "hex_mlp_lr_final", 0.00001),
            lr_delay_mult=0.01,
            max_steps=_max_steps,
        )
    
    def update_learning_rate(self, iteration: int) -> Optional[float]:
        """Update learning rates for all student models."""
        plane_lr = self._plane_lr_func(iteration)
        mlp_lr = self._mlp_lr_func(iteration)
        
        for param_group in self.optimizer.param_groups:
            if "planes" in param_group["name"]:
                param_group["lr"] = plane_lr
            elif "mlp" in param_group["name"]:
                param_group["lr"] = mlp_lr
        
        return plane_lr
    
    def get_distillation_loss(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss: student vs teacher predictions."""
        with torch.no_grad():
            teacher_d_xyz, teacher_d_rot, teacher_d_scale = self.step_teacher(xyz, time_emb)
        
        student_d_xyz, student_d_rot, student_d_scale = self.step(xyz, time_emb, cluster_ids)
        
        # L2 loss between teacher and student predictions
        loss_xyz = F.mse_loss(student_d_xyz, teacher_d_xyz)
        loss_rot = F.mse_loss(student_d_rot, teacher_d_rot)
        loss_scale = F.mse_loss(student_d_scale, teacher_d_scale)
        
        return loss_xyz + loss_rot + loss_scale
    
    def save_weights(self, model_path: str, iteration: int) -> None:
        """Save all student model weights.
        
        New format: deform/iteration_30000/deform_cluster_*.pth
        """
        deform_dir = os.path.join(model_path, "deform")
        iter_dir = os.path.join(deform_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        for cluster_id, student in enumerate(self.students):
            weights_path = os.path.join(iter_dir, f"deform_cluster_{cluster_id}.pth")
            torch.save(student.state_dict(), weights_path)
    
    def load_weights(self, model_path: str, iteration: int = -1) -> None:
        """Load all student model weights.
        
        Supports both new format (deform/iteration_*/deform_cluster_*.pth)
        and legacy format (deform_cluster_*/iteration_*/deform.pth).
        """
        deform_dir = os.path.join(model_path, "deform")
        
        # Try new format first
        if iteration == -1:
            # Search for max iteration in new format
            import re
            iter_pattern = re.compile(r"iteration_(\d+)")
            max_iter = -1
            if os.path.isdir(deform_dir):
                for dirname in os.listdir(deform_dir):
                    match = iter_pattern.match(dirname)
                    if match:
                        iter_num = int(match.group(1))
                        if iter_num > max_iter:
                            max_iter = iter_num
            loaded_iter = max_iter if max_iter >= 0 else 0
        else:
            loaded_iter = iteration
        
        # Check new format
        iter_dir = os.path.join(deform_dir, f"iteration_{loaded_iter}")
        if os.path.isdir(iter_dir):
            # New format exists
            for cluster_id in range(self.n_clusters):
                weights_path = os.path.join(iter_dir, f"deform_cluster_{cluster_id}.pth")
                if os.path.exists(weights_path):
                    self.students[cluster_id].load_state_dict(torch.load(weights_path))
            return
        
        # Fallback to legacy format
        for cluster_id in range(self.n_clusters):
            if iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(model_path, f"deform_cluster_{cluster_id}"))
            else:
                loaded_iter = iteration
            
            weights_path = os.path.join(
                model_path, f"deform_cluster_{cluster_id}/iteration_{loaded_iter}/deform.pth"
            )
            if os.path.exists(weights_path):
                self.students[cluster_id].load_state_dict(torch.load(weights_path))
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get TV and L1 regularization from all student models."""
        tv_loss = sum(student.get_plane_tv_loss() for student in self.students)
        l1_loss = sum(student.get_plane_l1_loss() for student in self.students)
        return 1e-3 * tv_loss + 1e-4 * l1_loss
