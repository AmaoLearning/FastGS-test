import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
from utils.hexplane_utils import HexPlaneDeformNetwork
import os
from typing import Optional, Sequence, Tuple, Dict, List
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
import logging

logger = logging.getLogger(__name__)


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
        _plane_lr_init = training_args.hex_plane_lr_init
        _plane_lr_final = training_args.hex_plane_lr_final
        _mlp_lr_init = training_args.hex_mlp_lr_init
        _mlp_lr_final = training_args.hex_mlp_lr_final

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

    def get_regularization_loss(
        self, tv_temporal_weight: Optional[float] = None
    ) -> torch.Tensor:
        """Get TV and L1 regularization.

        Parameters
        ----------
        tv_temporal_weight : float or None
            When not None, the temporal axis of XT/YT/ZT planes uses this
            weight instead of the default 1e-3, enabling temporal TV annealing.
        """
        if tv_temporal_weight is not None:
            spatial_tv, temporal_tv = self.deform.get_plane_tv_loss_split()
            l1 = self.deform.get_plane_l1_loss()
            return 1e-3 * spatial_tv + tv_temporal_weight * temporal_tv + 1e-4 * l1
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
    
    Capacity Allocation:
    - Supports per-cluster capacity configuration via student_configs list
    - Backward compatible: accepts uniform parameters (auto-converted to list)
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
        # Student config (supports per-cluster or uniform)
        student_configs: Optional[Sequence[Dict]] = None,  # List of per-cluster configs (new)
        # Legacy uniform parameters (backward compatible)
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
        # Support both per-cluster configs (new) and uniform configs (legacy)
        if student_configs is not None:
            # New mode: per-cluster capacity configuration
            if len(student_configs) != n_clusters:
                raise ValueError(f"student_configs length ({len(student_configs)}) must match n_clusters ({n_clusters})")
            self.student_configs = student_configs
        else:
            # Legacy mode: uniform configuration for all clusters
            # Convert to list format for unified processing
            self.student_configs = [
                {
                    "spatial_resolutions": student_spatial_resolutions,
                    "time_resolutions": student_time_resolutions,
                    "feat_dim": student_feat_dim,
                    "mlp_hidden_dim": student_mlp_hidden_dim,
                    "mlp_layer_num": student_mlp_num_hidden,  # Unified naming
                }
                for _ in range(n_clusters)
            ]
        
        # Create student networks
        self.students = nn.ModuleList()
        for cluster_id, config in enumerate(self.student_configs):
            student = HexPlaneDeformNetwork(
                spatial_resolutions=config["spatial_resolutions"],
                time_resolutions=config["time_resolutions"],
                feat_dim=config["feat_dim"],
                mlp_hidden_dim=config["mlp_hidden_dim"],
                mlp_num_hidden=config["mlp_layer_num"],
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
        
        # Parallel forward pass using CUDA streams for true async execution
        if len(cluster_inputs) == 0:
            return d_xyz, d_rotation, d_scaling
        
        # Use multiple CUDA streams to submit all student forward passes asynchronously
        # This enables kernel overlap and better GPU utilization
        n_streams = min(len(cluster_inputs), 4)  # Limit streams to avoid overhead
        streams = [torch.cuda.Stream(device=device) for _ in range(n_streams)]
        
        # Record CUDA events for synchronization
        events = []
        cluster_outputs = []
        
        # Submit all student forward passes on different streams
        for i, (student, (xyz_c, time_c)) in enumerate(zip(self.students, cluster_inputs)):
            stream_id = i % n_streams
            with torch.cuda.stream(streams[stream_id]):
                d_xyz_c, d_rotation_c, d_scaling_c = student(xyz_c, time_c)
                cluster_outputs.append((d_xyz_c, d_rotation_c, d_scaling_c))
                # Record event for this stream
                event = torch.cuda.Event()
                event.record(streams[stream_id])
                events.append(event)
        
        # Wait for all streams to complete
        for event in events:
            event.synchronize()
        
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
    
    def train_setting(self, training_args, start_iteration: int = 0) -> None:
        """Initialize optimizers for all student models.

        Args:
            training_args: OptimizationParams instance.
            start_iteration: The global iteration at which students are created.
                LR schedulers are built over the *remaining* steps so that
                students start from ``lr_init`` and decay to ``lr_final`` by
                ``deform_lr_max_steps``.
        """
        self._lr_start_iter = start_iteration
        param_groups = []
        
        for cluster_id, student in enumerate(self.students):
            plane_params = list(student.hexplane.parameters())
            mlp_params = list(student.decoder.parameters())
            pe_params = (
                list(student.pe_xyz.parameters())
                + list(student.pe_t.parameters())
            )
            
            _plane_lr_init = training_args.hex_plane_lr_init
            _plane_lr_final = training_args.hex_plane_lr_final
            _mlp_lr_init = training_args.hex_mlp_lr_init
            _mlp_lr_final = training_args.hex_mlp_lr_final
            
            param_groups.extend([
                {"params": plane_params, "lr": _plane_lr_init, "name": f"student_{cluster_id}_planes"},
                {"params": mlp_params + pe_params, "lr": _mlp_lr_init, "name": f"student_{cluster_id}_mlp"},
            ])
        
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
        # LR schedulers — use *remaining* steps so students start from lr_init.
        _remaining_steps = max(training_args.deform_lr_max_steps - start_iteration, 1)
        self._plane_lr_func = get_expon_lr_func(
            lr_init=training_args.hex_plane_lr_init,
            lr_final=training_args.hex_plane_lr_final,
            lr_delay_mult=0.01,
            max_steps=_remaining_steps,
        )
        self._mlp_lr_func = get_expon_lr_func(
            lr_init=training_args.hex_mlp_lr_init,
            lr_final=training_args.hex_mlp_lr_final,
            lr_delay_mult=0.01,
            max_steps=_remaining_steps,
        )
        logger.info(
            "[ClusteredDeform] LR scheduler: start_iter=%d, remaining_steps=%d, "
            "plane_lr_init=%.4f→%.4f, mlp_lr_init=%.6f→%.6f",
            start_iteration, _remaining_steps,
            training_args.hex_plane_lr_init, training_args.hex_plane_lr_final,
            training_args.hex_mlp_lr_init, training_args.hex_mlp_lr_final,
        )
    
    def update_learning_rate(self, iteration: int) -> Optional[float]:
        """Update learning rates for all student models.

        Uses the *relative* iteration count (iteration - start_iteration)
        so that students begin training from ``lr_init`` regardless of when
        they are created during the global training loop.
        """
        relative_iter = max(iteration - self._lr_start_iter, 0)
        plane_lr = self._plane_lr_func(relative_iter)
        mlp_lr = self._mlp_lr_func(relative_iter)
        
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
        student_d_xyz: Optional[torch.Tensor] = None,
        student_d_rot: Optional[torch.Tensor] = None,
        student_d_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss: student vs teacher predictions.
        
        Args:
            xyz: Gaussian positions (N, 3)
            time_emb: Time embeddings (N, 1)
            cluster_ids: Cluster assignments (N,)
            student_d_xyz: Pre-computed student deformation (N, 3). If None, recomputes.
            student_d_rot: Pre-computed student rotation (N, 4). If None, recomputes.
            student_d_scale: Pre-computed student scaling (N, 3). If None, recomputes.
        
        Returns:
            Distillation loss (scalar)
        """
        # Teacher prediction (no_grad, frozen)
        with torch.no_grad():
            teacher_d_xyz, teacher_d_rot, teacher_d_scale = self.step_teacher(xyz, time_emb)
        
        # Reuse pre-computed student predictions if provided, otherwise compute fresh
        if student_d_xyz is None or student_d_rot is None or student_d_scale is None:
            student_d_xyz, student_d_rot, student_d_scale = self.step(xyz, time_emb, cluster_ids)
        
        # L2 loss between teacher and student predictions
        loss_xyz = F.mse_loss(student_d_xyz, teacher_d_xyz)
        loss_rot = F.mse_loss(student_d_rot, teacher_d_rot)
        loss_scale = F.mse_loss(student_d_scale, teacher_d_scale)
        
        return loss_xyz + loss_rot + loss_scale
    
    def save_weights(self, model_path: str, iteration: int) -> None:
        """Save all student model weights with tier labels from student_configs.
        
        Naming convention:
        - Single-tier (tiered/linear strategy):
            ``deform_cluster_{tier}_{id}.pth``
        - Dual-tier (frequency strategy with independent HexPlane/MLP tiers):
            ``deform_cluster_hex{hex_tier}_mlp{mlp_tier}_{id}.pth``
        
        The tier labels are read directly from each student's configuration.
        
        Args:
            model_path: Base model path.
            iteration: Current iteration number.
        """
        deform_dir = os.path.join(model_path, "deform")
        iter_dir = os.path.join(deform_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Read tier labels directly from student_configs
        for cluster_id, student in enumerate(self.students):
            if cluster_id < len(self.student_configs):
                config = self.student_configs[cluster_id]
                hex_tier = config.get("hex_tier", None)
                mlp_tier = config.get("mlp_tier", None)
                
                if hex_tier is not None and mlp_tier is not None:
                    # Dual-tier naming for frequency-based allocation
                    fname = f"deform_cluster_hex{hex_tier}_mlp{mlp_tier}_{cluster_id}.pth"
                else:
                    # Single-tier naming for tiered/linear allocation
                    tier = config.get("tier", "unknown")
                    fname = f"deform_cluster_{tier}_{cluster_id}.pth"
                
                weights_path = os.path.join(iter_dir, fname)
            else:
                # Fallback to legacy format if config missing
                weights_path = os.path.join(iter_dir, f"deform_cluster_{cluster_id}.pth")
                fname = os.path.basename(weights_path)
            
            torch.save(student.state_dict(), weights_path)
            print(f"[INFO] Saved cluster {cluster_id} → {fname}")
    
    def load_weights(self, model_path: str, iteration: int = -1) -> None:
        """Load all student model weights with tier-based naming.
        
        Supports two naming conventions:
        - Single-tier: ``deform_cluster_{tier}_{id}.pth``
        - Dual-tier:   ``deform_cluster_hex{hex_tier}_mlp{mlp_tier}_{id}.pth``
        
        Args:
            model_path: Base model path.
            iteration: Iteration to load (-1 for latest).
        
        Raises:
            FileNotFoundError: If no weight files are found.
        """
        deform_dir = os.path.join(model_path, "deform")
        
        # Find iteration directory
        if iteration == -1:
            # Search for max iteration
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
        
        iter_dir = os.path.join(deform_dir, f"iteration_{loaded_iter}")
        if not os.path.isdir(iter_dir):
            raise FileNotFoundError(f"Cannot find deform weights directory at {iter_dir}")
        
        # Scan directory for weight files (both single-tier and dual-tier)
        import re
        # Dual-tier: deform_cluster_hex{H}_mlp{M}_{id}.pth
        dual_pattern = re.compile(
            r"deform_cluster_hex(?P<hex_tier>high|medium|low)_mlp(?P<mlp_tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
        )
        # Single-tier: deform_cluster_{tier}_{id}.pth
        single_pattern = re.compile(
            r"deform_cluster_(?P<tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
        )
        
        tier_files: Dict[int, Dict] = {}  # {cluster_id: {"filepath": ..., ...}}
        
        print(f"[INFO] Scanning {iter_dir} for student model weights...")
        
        for filename in sorted(os.listdir(iter_dir)):
            # Try dual-tier first (more specific pattern)
            m = dual_pattern.match(filename)
            if m:
                cid = int(m.group("cluster_id"))
                tier_files[cid] = {
                    "filepath": os.path.join(iter_dir, filename),
                    "hex_tier": m.group("hex_tier"),
                    "mlp_tier": m.group("mlp_tier"),
                }
                print(f"[INFO] Found: {filename} (cluster {cid}, hex={m.group('hex_tier')}, mlp={m.group('mlp_tier')})")
                continue
            # Fall back to single-tier
            m = single_pattern.match(filename)
            if m:
                cid = int(m.group("cluster_id"))
                tier_files[cid] = {
                    "filepath": os.path.join(iter_dir, filename),
                    "tier": m.group("tier"),
                }
                print(f"[INFO] Found: {filename} (cluster {cid}, tier={m.group('tier')})")
        
        if not tier_files:
            raise FileNotFoundError(
                f"No student weight files found in {iter_dir}. "
                f"Expected: deform_cluster_{{tier}}_{{id}}.pth or "
                f"deform_cluster_hex{{H}}_mlp{{M}}_{{id}}.pth"
            )
        
        # Load weights into student models
        print(f"[INFO] Loading {len(tier_files)} student models...")
        loaded_count = 0
        
        for cluster_id in sorted(tier_files.keys()):
            if cluster_id >= len(self.students):
                print(f"[WARNING] Cluster {cluster_id} exceeds student count ({len(self.students)}), skipping")
                continue
            
            info = tier_files[cluster_id]
            filepath = info["filepath"]
            self.students[cluster_id].load_state_dict(torch.load(filepath))
            loaded_count += 1
            
            if "hex_tier" in info:
                print(f"[INFO] Loaded cluster {cluster_id} (hex={info['hex_tier']}, mlp={info['mlp_tier']})")
            else:
                print(f"[INFO] Loaded cluster {cluster_id} (tier={info['tier']})")
        
        if loaded_count == 0:
            raise FileNotFoundError(f"Failed to load any student models from {iter_dir}")
        
        print(f"[INFO] Successfully loaded {loaded_count} student models")
    
    def initialize_students_with_warm_init(
        self,
        warm_init_cfg,
        noise_std_per_student: Optional[List[float]] = None,
    ) -> None:
        """
        使用热启动初始化所有学生网络。
        
        在第 15000 轮聚类完成后调用，将教师 HexPlane 参数降采样迁移到学生。
        
        Args:
            warm_init_cfg: WarmInitConfig 实例，包含热启动配置。
            noise_std_per_student: 可选，每个学生的噪声标准差 (打破对称性)。
                若为 None，则使用 warm_init_cfg.noise_std 统一值。
        """
        if not warm_init_cfg.enabled:
            logger.info("WarmInit 已禁用，跳过学生初始化。")
            return
        
        # 确保教师处于 eval 模式且参数冻结
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        
        # 为不同学生设置递增噪声 (打破对称性)
        n = len(self.students)
        if noise_std_per_student is None:
            base_noise = warm_init_cfg.noise_std
            noise_std_per_student = [
                base_noise * (1.0 + 0.5 * i / max(n - 1, 1))
                for i in range(n)
            ]

        from utils.warm_init_utils import warm_init_all_students
        
        # 执行热启动初始化
        warm_init_all_students(
            teacher_network=self.teacher,
            student_networks=list(self.students),
            student_configs=self.student_configs,
            cfg=warm_init_cfg,
            noise_std_per_student=noise_std_per_student,
        )
        
        logger.info(f"所有 {n} 个学生网络热启动初始化完成。")
    
    def get_regularization_loss(
        self, tv_temporal_weight: Optional[float] = None
    ) -> torch.Tensor:
        """Get TV and L1 regularization from all student models.

        Parameters
        ----------
        tv_temporal_weight : float or None
            When not None, temporal axes of XT/YT/ZT planes use this weight.
        """
        if tv_temporal_weight is not None:
            spatial_tv = sum(s.get_plane_tv_loss_split()[0] for s in self.students)
            temporal_tv = sum(s.get_plane_tv_loss_split()[1] for s in self.students)
            l1 = sum(s.get_plane_l1_loss() for s in self.students)
            return 1e-3 * spatial_tv + tv_temporal_weight * temporal_tv + 1e-4 * l1
        tv_loss = sum(student.get_plane_tv_loss() for student in self.students)
        l1_loss = sum(student.get_plane_l1_loss() for student in self.students)
        return 1e-3 * tv_loss + 1e-4 * l1_loss

    def get_per_student_regularization_losses(
        self, tv_temporal_weight: Optional[float] = None
    ) -> List[torch.Tensor]:
        """Return per-student regularization losses for parallel backward.

        Each element is the regularization loss for student *k*.
        The losses are **not** summed so callers can backward them on
        independent CUDA streams.

        Parameters
        ----------
        tv_temporal_weight : float or None
            When not None, temporal axes of XT/YT/ZT planes use this weight.
        """
        losses: List[torch.Tensor] = []
        for student in self.students:
            if tv_temporal_weight is not None:
                s_tv, t_tv = student.get_plane_tv_loss_split()
                l1 = student.get_plane_l1_loss()
                reg_k = 1e-3 * s_tv + tv_temporal_weight * t_tv + 1e-4 * l1
            else:
                reg_k = 1e-3 * student.get_plane_tv_loss() + 1e-4 * student.get_plane_l1_loss()
            losses.append(reg_k)
        return losses
