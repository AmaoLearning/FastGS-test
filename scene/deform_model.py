import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
from utils.hexplane_utils import (
    HexPlaneDeformNetwork,
    BatchedHexPlaneDeformNetwork,
)
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


class ParallelBackwardHandle:
    """Gradient handoff handle for ClusteredDeformModel parallel backward.

    ``ClusteredDeformModel.step(return_handoffs=True)`` returns this object
    alongside the assembled deformation tensors.  The assembled d_xyz /
    d_rotation / d_scaling are built from *leaf* tensors (detached student
    outputs re-wrapped with ``requires_grad=True``), so ``loss.backward()``
    propagates only through the shallow render graph and stops at these
    leaves — never entering the student networks.

    After ``loss.backward()``, call :meth:`backward_parallel` to propagate
    the leaf gradients into the student network parameters concurrently,
    using one CUDA stream per student and Python threading so that GPU
    kernels from independent networks execute in parallel on the same device.
    """

    def __init__(
        self,
        raw_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        leaf_tensors: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> None:
        # raw_outputs[k] = (xyz_raw, rot_raw, scale_raw) — student outputs with grad_fn
        # leaf_tensors[k] = (xyz_leaf, rot_leaf, scale_leaf) — detached leaves,
        #   require_grad=True, used in the assembled render inputs
        self._raw = raw_outputs
        self._leaves = leaf_tensors
        self._device = device

    def backward_parallel(self) -> None:
        """Propagate gradients from handoff leaves into student network parameters.

        Must be called **after** ``loss.backward()`` has populated
        ``leaf.grad`` for every handoff leaf.

        Each student's backward runs on its own CUDA stream inside a Python
        thread.  Because the student computation graphs share no parameters,
        the autograd engine can execute their CUDA kernels concurrently.

        Synchronisation guarantee
        -------------------------
        A CUDA event is recorded on the default stream immediately before
        the threads are started.  Each worker stream waits on this event so
        that ``leaf.grad`` data written by ``loss.backward()`` is visible
        before the student backward kernels read it.
        """
        n = len(self._raw)
        if n == 0:
            return

        streams = [torch.cuda.Stream(device=self._device) for _ in range(n)]

        # Mark the point at which loss.backward() finished on the default stream.
        # Worker streams will wait for this before reading leaf.grad data.
        backward_done = torch.cuda.Event()
        backward_done.record()  # recorded on the current (default) stream

        def _backward_one(idx: int) -> None:
            xyz_raw, rot_raw, scale_raw = self._raw[idx]
            xyz_leaf, rot_leaf, scale_leaf = self._leaves[idx]
            stream = streams[idx]

            # Collect tensors/grads for a single torch.autograd.backward call
            # so the student's activations are traversed only once.
            tensors: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            for raw, leaf in (
                (xyz_raw, xyz_leaf),
                (rot_raw, rot_leaf),
                (scale_raw, scale_leaf),
            ):
                if leaf.grad is not None:
                    tensors.append(raw)
                    grads.append(leaf.grad)

            if not tensors:
                return

            with torch.cuda.stream(stream):
                # GPU-level barrier: wait until loss.backward() grads are ready.
                stream.wait_event(backward_done)
                torch.autograd.backward(tensors, grads)

        threads = [threading.Thread(target=_backward_one, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Bring all worker streams back in sync with the default stream so that
        # optimizer.step() observes all accumulated parameter gradients.
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)


class ClusteredDeformModel:
    """Multi-deform model for clustered dynamic Gaussians.
    
    This class manages multiple deformation fields, each responsible for a cluster
    of dynamic Gaussians. Uses per-cluster student HexPlane networks initialized
    teacher model to multiple student models.
    
    Architecture:
    - 1 teacher deform field (original 4DGS model, frozen after clustering)
    - K student deform fields (one per cluster, configurable capacity)
    - Student model architecture is manually configurable
    
    Training objectives:
    1. Visual quality loss (L1 + SSIM) - same as original
    
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
        # ── Batched mode ─────────────────────────────────────────────
        use_batched_students: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.is_blender = is_blender
        self.is_6dof = is_6dof
        self.fusion = fusion
        self.use_batched_students = use_batched_students
        
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
        self.students: Optional[nn.ModuleList] = None
        self.batched_net: Optional[BatchedHexPlaneDeformNetwork] = None

        if use_batched_students:
            # ── Batched path (Path A + B) ────────────────────────────────
            # All students must have identical architecture.
            # Verify uniformity and extract common config.
            _cfg0 = self.student_configs[0]
            _uniform = all(
                c["spatial_resolutions"] == _cfg0["spatial_resolutions"]
                and c["time_resolutions"] == _cfg0["time_resolutions"]
                and c["feat_dim"] == _cfg0["feat_dim"]
                and c["mlp_hidden_dim"] == _cfg0["mlp_hidden_dim"]
                and c.get("mlp_layer_num", 2) == _cfg0.get("mlp_layer_num", 2)
                for c in self.student_configs
            )
            if not _uniform:
                raise ValueError(
                    "use_batched_students=True requires all student_configs to be "
                    "identical (uniform architecture). Found heterogeneous configs."
                )
            self.batched_net = BatchedHexPlaneDeformNetwork(
                K=n_clusters,
                spatial_resolutions=_cfg0["spatial_resolutions"],
                time_resolutions=_cfg0["time_resolutions"],
                feat_dim=_cfg0["feat_dim"],
                mlp_hidden_dim=_cfg0["mlp_hidden_dim"],
                mlp_num_hidden=_cfg0.get("mlp_layer_num", 2),
                fusion=fusion,
                is_blender=is_blender,
                is_6dof=is_6dof,
            ).cuda()  # move all parameters AND registered buffers (aabb_mins/maxs) to GPU
            logger.info(
                "[ClusteredDeform] Batched mode: K=%d, spatial=%s, feat_dim=%d, "
                "mlp_hidden=%d — grid_sample calls: %d (constant w.r.t. K)",
                n_clusters,
                _cfg0["spatial_resolutions"],
                _cfg0["feat_dim"],
                _cfg0["mlp_hidden_dim"],
                len(_cfg0["spatial_resolutions"]) * 6,
            )
        else:
            # ── Sequential path (original behaviour) ────────────────────
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
        """Set global AABB for teacher + all students."""
        self.teacher.set_aabb(points, padding=padding)
        if self.use_batched_students:
            self.batched_net.set_aabb_all(points, padding=padding)
        else:
            for student in self.students:
                student.set_aabb(points, padding=padding)

    def set_per_cluster_aabb(
        self,
        points: torch.Tensor,
        cluster_labels: torch.Tensor,
        padding: float = 0.15,
        deform_max: Optional[torch.Tensor] = None,
        deform_min: Optional[torch.Tensor] = None,
    ) -> None:
        """Set per-cluster AABB for each student network based on canonical positions.

        The deformation field is always queried at the Gaussian's **canonical
        (static) position** — never at the deformed position.  Therefore the
        AABB only needs to tightly cover the canonical positions of Gaussians
        belonging to each cluster, plus a proportional padding margin:

        .. code-block:: text

            aabb_min_k = static_min_k − padding × extent_k
            aabb_max_k = static_max_k + padding × extent_k

        The ``deform_max`` / ``deform_min`` parameters are accepted for
        backward compatibility but are intentionally **ignored**.
        Displacement-aware AABB expansion caused adjacent clusters to overlap
        by 25–50 % in high-displacement scenes, which wasted grid capacity and
        distorted the TV regularisation's physical-distance semantics.

        The teacher always receives the **global** AABB (it handles all
        clusters uniformly).

        Parameters
        ----------
        points : Tensor ``(N, 3)``
            Current Gaussian canonical positions (detached from the graph).
        cluster_labels : Tensor ``(N,)`` int
            Per-Gaussian cluster index; −1 marks static Gaussians (ignored).
        padding : float
            Fraction of cluster extent added as safety margin on each side.
        deform_max : Tensor ``(N, 3)`` or None
            Ignored.  Kept for API compatibility.
        deform_min : Tensor ``(N, 3)`` or None
            Ignored.  Kept for API compatibility.
        """
        # Teacher always uses the full-scene AABB.
        self.teacher.set_aabb(points, padding=padding)

        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            n_pts = int(mask.sum().item())

            if n_pts < 4:
                if self.use_batched_students:
                    self.batched_net.set_aabb_single(k, points, padding=padding)
                else:
                    self.students[k].set_aabb(points, padding=padding)
                logger.warning(
                    "[ClusteredDeform] Cluster %d has only %d Gaussians — "
                    "using global AABB as fallback.", k, n_pts
                )
                continue

            pts_k = points[mask]
            static_min_k = pts_k.min(dim=0).values
            static_max_k = pts_k.max(dim=0).values
            static_extent_k = (static_max_k - static_min_k).clamp(min=1e-6)

            # Use canonical (static) positions only — no displacement expansion.
            aabb_min = static_min_k - padding * static_extent_k
            aabb_max = static_max_k + padding * static_extent_k

            if self.use_batched_students:
                self.batched_net.aabb_mins[k] = aabb_min
                self.batched_net.aabb_maxs[k] = aabb_max
            else:
                self.students[k].aabb_min.copy_(aabb_min)
                self.students[k].aabb_max.copy_(aabb_max)

            logger.debug(
                "[ClusteredDeform] Student %d AABB: min=[%.3f,%.3f,%.3f] "
                "max=[%.3f,%.3f,%.3f] (n_pts=%d, canonical-only)",
                k,
                aabb_min[0].item(), aabb_min[1].item(), aabb_min[2].item(),
                aabb_max[0].item(), aabb_max[1].item(), aabb_max[2].item(),
                n_pts,
            )
    
    def set_cluster_labels(self, cluster_labels: torch.Tensor) -> None:
        """Set cluster labels for Gaussian assignment."""
        self._cluster_labels = cluster_labels  # (N,) int32, -1 for static
    
    def _step_impl(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
        return_handoffs: bool = False,
    ) -> Tuple:
        """Dispatch to batched or sequential forward pass."""
        if self.use_batched_students:
            if return_handoffs:
                raise ValueError(
                    "return_handoffs=True is incompatible with use_batched_students=True. "
                    "Batched autograd handles gradients automatically — "
                    "ParallelBackwardHandle is not needed."
                )
            return self._step_impl_batched(xyz, time_emb, cluster_ids)
        return self._step_impl_sequential(xyz, time_emb, cluster_ids, return_handoffs)

    def _step_impl_batched(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched forward pass using BatchedHexPlaneDeformNetwork.

        Kernel count is O(1) w.r.t. K (18 grid_sample + 2 bmm), compared to
        O(K * 18) for the sequential path.
        """
        if cluster_ids is None:
            cluster_ids = self._cluster_labels
        if cluster_ids is None:
            raise ValueError("No cluster labels provided. Call set_cluster_labels first.")

        # _cluster_labels is typically stored on CPU; move to xyz's device so
        # boolean masks derived from it are on the same device as d_xyz/d_rotation/d_scaling.
        device = xyz.device
        cluster_ids = cluster_ids.to(device)

        N = xyz.shape[0]
        dtype = xyz.dtype
        use_per_point_time = time_emb.shape[0] == N

        # Gather per-cluster inputs and masks
        cluster_xyz: List[torch.Tensor] = []
        cluster_t: List[torch.Tensor] = []
        cluster_counts: List[int] = []
        cluster_masks: List[torch.Tensor] = []

        for k in range(self.n_clusters):
            mask = (cluster_ids == k)
            count = int(mask.sum().item())
            cluster_masks.append(mask)
            cluster_counts.append(count)
            if count > 0:
                cluster_xyz.append(xyz[mask])
                cluster_t.append(
                    time_emb[mask] if use_per_point_time
                    else time_emb.expand(count, -1)
                )
            else:
                cluster_xyz.append(xyz.new_empty(0, 3))
                cluster_t.append(time_emb.new_empty(0, 1))

        # Batched forward → (K, N_max, 10)
        out = self.batched_net(cluster_xyz, cluster_t, cluster_counts)

        # Unpack into (N, 3), (N, 4), (N, 3)
        d_xyz = torch.zeros(N, 3, device=device, dtype=dtype)
        d_rotation = torch.zeros(N, 4, device=device, dtype=dtype)
        d_scaling = torch.zeros(N, 3, device=device, dtype=dtype)

        for k in range(self.n_clusters):
            n = cluster_counts[k]
            if n > 0:
                mask = cluster_masks[k]
                d_xyz[mask] = out[k, :n, :3]
                d_rotation[mask] = out[k, :n, 3:7]
                d_scaling[mask] = out[k, :n, 7:10]

        return d_xyz, d_rotation, d_scaling

    def _step_impl_sequential(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
        return_handoffs: bool = False,
    ) -> Tuple:
        """Forward pass for clustered deformation (parallel inference).
        
        Uses batched parallel inference: all student models receive their
        respective cluster inputs in a single concatenated batch, enabling
        CUDA kernel fusion and overlapping memory operations.
        
        Args:
            xyz: Gaussian positions (N, 3)
            time_emb: Time embeddings (N, 1) or (1, 1)
            cluster_ids: Cluster assignments for each Gaussian (N,) or None to use self._cluster_labels
            return_handoffs: If True, returns a ParallelBackwardHandle as a 4th element.
                The assembled d_xyz/d_rotation/d_scaling will be built from *leaf* tensors
                so that loss.backward() stops at the handoff boundary (shallow/fast).
                Call handle.backward_parallel() after loss.backward() to propagate
                gradients into student parameters concurrently on separate CUDA streams.
        
        Returns:
            (d_xyz, d_rotation, d_scaling) when return_handoffs=False (default).
            (d_xyz, d_rotation, d_scaling, handle) when return_handoffs=True.
        """
        if cluster_ids is None:
            cluster_ids = self._cluster_labels
        
        if cluster_ids is None:
            raise ValueError("No cluster labels provided. Set cluster_labels first.")
        
        N = xyz.shape[0]
        device = xyz.device
        dtype = xyz.dtype
        
        # Determine if time_emb needs per-point indexing
        use_per_point_time = time_emb.shape[0] == N
        
        # Gather inputs for all clusters in a single pass
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
            d_xyz = torch.zeros(N, 3, device=device, dtype=dtype)
            d_rotation = torch.zeros(N, 4, device=device, dtype=dtype)
            d_scaling = torch.zeros(N, 3, device=device, dtype=dtype)
            if return_handoffs:
                return d_xyz, d_rotation, d_scaling, ParallelBackwardHandle([], [], device)
            return d_xyz, d_rotation, d_scaling
        
        # Use multiple CUDA streams to submit all student forward passes asynchronously
        n_fwd_streams = min(len(cluster_inputs), 4)
        fwd_streams = [torch.cuda.Stream(device=device) for _ in range(n_fwd_streams)]
        
        fwd_events = []
        cluster_outputs = []  # [(xyz_raw, rot_raw, scale_raw), ...]
        
        for i, (student, (xyz_c, time_c)) in enumerate(zip(self.students, cluster_inputs)):
            stream_id = i % n_fwd_streams
            with torch.cuda.stream(fwd_streams[stream_id]):
                d_xyz_c, d_rotation_c, d_scaling_c = student(xyz_c, time_c)
                cluster_outputs.append((d_xyz_c, d_rotation_c, d_scaling_c))
                event = torch.cuda.Event()
                event.record(fwd_streams[stream_id])
                fwd_events.append(event)
        
        # Wait for all forward passes to complete
        for event in fwd_events:
            event.synchronize()
        
        if not return_handoffs:
            # ── Original path: assemble directly from student outputs ──────────
            d_xyz = torch.zeros(N, 3, device=device, dtype=dtype)
            d_rotation = torch.zeros(N, 4, device=device, dtype=dtype)
            d_scaling = torch.zeros(N, 3, device=device, dtype=dtype)
            for i, (d_xyz_c, d_rotation_c, d_scaling_c) in enumerate(cluster_outputs):
                mask = cluster_masks[i]
                d_xyz[mask] = d_xyz_c
                d_rotation[mask] = d_rotation_c
                d_scaling[mask] = d_scaling_c
            return d_xyz, d_rotation, d_scaling

        # ── Handoff path: assemble from *leaf* tensors ─────────────────────────
        # Each leaf is a detached copy of the student output re-wrapped with
        # requires_grad=True.  The assembled render inputs depend on these leaves,
        # NOT on the student computation graphs.  Therefore loss.backward() only
        # traverses the render graph (shallow/fast) and populates leaf.grad.
        # handle.backward_parallel() then fans out the student network backward
        # passes concurrently using CUDA streams + Python threads.

        leaf_tensors: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        xyz_leaf_list: List[torch.Tensor] = []
        rot_leaf_list: List[torch.Tensor] = []
        scale_leaf_list: List[torch.Tensor] = []
        index_list: List[torch.Tensor] = []

        for i, (d_xyz_c, d_rotation_c, d_scaling_c) in enumerate(cluster_outputs):
            # Detach and re-attach as leaf so autograd stops here during loss.backward()
            xyz_leaf = d_xyz_c.detach().requires_grad_(True)
            rot_leaf = d_rotation_c.detach().requires_grad_(True)
            scale_leaf = d_scaling_c.detach().requires_grad_(True)
            leaf_tensors.append((xyz_leaf, rot_leaf, scale_leaf))
            xyz_leaf_list.append(xyz_leaf)
            rot_leaf_list.append(rot_leaf)
            scale_leaf_list.append(scale_leaf)
            index_list.append(cluster_masks[i].nonzero(as_tuple=True)[0])

        # Assemble via a single index_put each — efficient and correctly tracked.
        # torch.cat backward distributes leaf gradients back to each piece.
        all_indices = torch.cat(index_list)
        d_xyz = torch.zeros(N, 3, device=device, dtype=dtype).index_put(
            (all_indices,), torch.cat(xyz_leaf_list)
        )
        d_rotation = torch.zeros(N, 4, device=device, dtype=dtype).index_put(
            (all_indices,), torch.cat(rot_leaf_list)
        )
        d_scaling = torch.zeros(N, 3, device=device, dtype=dtype).index_put(
            (all_indices,), torch.cat(scale_leaf_list)
        )

        handle = ParallelBackwardHandle(
            raw_outputs=cluster_outputs,
            leaf_tensors=leaf_tensors,
            device=device,
        )
        return d_xyz, d_rotation, d_scaling, handle

    def step(
        self,
        xyz: torch.Tensor,
        time_emb: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
        return_handoffs: bool = False,
    ) -> Tuple:
        """Forward pass for clustered deformation.

        Delegates to :meth:`_step_impl`.  See that method for full docs.
        """
        return self._step_impl(xyz, time_emb, cluster_ids=cluster_ids, return_handoffs=return_handoffs)

    def train_setting(self, training_args, start_iteration: int = 0) -> None:
        """Initialize optimizers for all student models."""
        self._lr_start_iter = start_iteration
        _plane_lr_init = training_args.hex_plane_lr_init
        _plane_lr_final = training_args.hex_plane_lr_final
        _mlp_lr_init = training_args.hex_mlp_lr_init
        _mlp_lr_final = training_args.hex_mlp_lr_final

        if self.use_batched_students:
            # Batched path: 2 param groups total (planes + mlp), not 2*K
            param_groups = [
                {
                    "params": list(self.batched_net.hexplane.parameters()),
                    "lr": _plane_lr_init,
                    "name": "students_planes",
                },
                {
                    "params": (
                        list(self.batched_net.decoder.parameters())
                        + list(self.batched_net.pe_xyz.parameters())
                        + list(self.batched_net.pe_t.parameters())
                    ),
                    "lr": _mlp_lr_init,
                    "name": "students_mlp",
                },
            ]
        else:
            # Sequential path: 2 param groups per student (original behaviour)
            param_groups = []
            for cluster_id, student in enumerate(self.students):
                param_groups.extend([
                    {
                        "params": list(student.hexplane.parameters()),
                        "lr": _plane_lr_init,
                        "name": f"student_{cluster_id}_planes",
                    },
                    {
                        "params": (
                            list(student.decoder.parameters())
                            + list(student.pe_xyz.parameters())
                            + list(student.pe_t.parameters())
                        ),
                        "lr": _mlp_lr_init,
                        "name": f"student_{cluster_id}_mlp",
                    },
                ])

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        _remaining_steps = max(training_args.deform_lr_max_steps - start_iteration, 1)
        self._plane_lr_func = get_expon_lr_func(
            lr_init=_plane_lr_init, lr_final=_plane_lr_final,
            lr_delay_mult=0.01, max_steps=_remaining_steps,
        )
        self._mlp_lr_func = get_expon_lr_func(
            lr_init=_mlp_lr_init, lr_final=_mlp_lr_final,
            lr_delay_mult=0.01, max_steps=_remaining_steps,
        )
        logger.info(
            "[ClusteredDeform] LR scheduler: start_iter=%d, remaining_steps=%d, "
            "plane_lr=%.4f→%.4f, mlp_lr=%.6f→%.6f, batched=%s",
            start_iteration, _remaining_steps,
            _plane_lr_init, _plane_lr_final,
            _mlp_lr_init, _mlp_lr_final,
            self.use_batched_students,
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
    
    def save_weights(self, model_path: str, iteration: int) -> None:
        """Save student model weights."""
        iter_dir = os.path.join(model_path, "deform", f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        if self.use_batched_students:
            # Batched path: single file for all K students
            weights_path = os.path.join(iter_dir, "batched_students.pth")
            torch.save(self.batched_net.state_dict(), weights_path)
            print(f"[INFO] Saved batched students (K={self.n_clusters}) → batched_students.pth")
            return

        # Sequential path: one file per student
        for cluster_id, student in enumerate(self.students):
            if cluster_id < len(self.student_configs):
                config = self.student_configs[cluster_id]
                hex_tier = config.get("hex_tier", None)
                mlp_tier = config.get("mlp_tier", None)
                if hex_tier is not None and mlp_tier is not None:
                    fname = f"deform_cluster_hex{hex_tier}_mlp{mlp_tier}_{cluster_id}.pth"
                else:
                    tier = config.get("tier", "unknown")
                    fname = f"deform_cluster_{tier}_{cluster_id}.pth"
            else:
                fname = f"deform_cluster_{cluster_id}.pth"
            weights_path = os.path.join(iter_dir, fname)
            torch.save(student.state_dict(), weights_path)
            print(f"[INFO] Saved cluster {cluster_id} → {fname}")
    
    def load_weights(self, model_path: str, iteration: int = -1) -> None:
        """Load student model weights."""
        deform_dir = os.path.join(model_path, "deform")

        if iteration == -1:
            import re
            iter_pattern = re.compile(r"iteration_(\d+)")
            max_iter = -1
            if os.path.isdir(deform_dir):
                for dirname in os.listdir(deform_dir):
                    m = iter_pattern.match(dirname)
                    if m:
                        max_iter = max(max_iter, int(m.group(1)))
            loaded_iter = max_iter if max_iter >= 0 else 0
        else:
            loaded_iter = iteration

        iter_dir = os.path.join(deform_dir, f"iteration_{loaded_iter}")
        if not os.path.isdir(iter_dir):
            raise FileNotFoundError(f"Cannot find deform weights directory at {iter_dir}")

        if self.use_batched_students:
            weights_path = os.path.join(iter_dir, "batched_students.pth")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"Batched weights not found: {weights_path}. "
                    "If loading from a sequential checkpoint, use use_batched_students=False."
                )
            self.batched_net.load_state_dict(torch.load(weights_path, map_location="cuda"))
            print(f"[INFO] Loaded batched students (K={self.n_clusters}) from {weights_path}")
            return

        # Sequential path
        import re
        dual_pattern = re.compile(
            r"deform_cluster_hex(?P<hex_tier>high|medium|low)_mlp(?P<mlp_tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
        )
        single_pattern = re.compile(
            r"deform_cluster_(?P<tier>high|medium|low)_(?P<cluster_id>\d+)\.pth"
        )

        tier_files: Dict[int, Dict] = {}
        for filename in sorted(os.listdir(iter_dir)):
            m = dual_pattern.match(filename)
            if m:
                cid = int(m.group("cluster_id"))
                tier_files[cid] = {
                    "filepath": os.path.join(iter_dir, filename),
                    "hex_tier": m.group("hex_tier"), "mlp_tier": m.group("mlp_tier"),
                }
                continue
            m = single_pattern.match(filename)
            if m:
                cid = int(m.group("cluster_id"))
                tier_files[cid] = {
                    "filepath": os.path.join(iter_dir, filename),
                    "tier": m.group("tier"),
                }

        if not tier_files:
            raise FileNotFoundError(f"No student weight files found in {iter_dir}")

        loaded_count = 0
        for cluster_id in sorted(tier_files.keys()):
            if cluster_id >= len(self.students):
                print(f"[WARNING] Cluster {cluster_id} exceeds student count, skipping")
                continue
            info = tier_files[cluster_id]
            self.students[cluster_id].load_state_dict(torch.load(info["filepath"]))
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
        """Initialize student networks from teacher via warm-init."""
        if not warm_init_cfg.enabled:
            logger.info("WarmInit disabled, skipping student initialization.")
            return

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        n = self.n_clusters
        if noise_std_per_student is None:
            base_noise = warm_init_cfg.noise_std
            noise_std_per_student = [
                base_noise * (1.0 + 0.5 * i / max(n - 1, 1))
                for i in range(n)
            ]

        if self.use_batched_students:
            # Transfer teacher planes into each student's slice of the batched grid.
            # warm_init_all_students operates on HexPlaneDeformNetwork instances, so
            # we reconstruct temporary single-student networks, warm-init them, then
            # copy their plane parameters back into self.batched_net.
            from utils.warm_init_utils import warm_init_all_students

            _cfg0 = self.student_configs[0]
            tmp_students: List[HexPlaneDeformNetwork] = []
            for _ in range(n):
                s = HexPlaneDeformNetwork(
                    spatial_resolutions=_cfg0["spatial_resolutions"],
                    time_resolutions=_cfg0["time_resolutions"],
                    feat_dim=_cfg0["feat_dim"],
                    mlp_hidden_dim=_cfg0["mlp_hidden_dim"],
                    mlp_num_hidden=_cfg0.get("mlp_layer_num", 2),
                    fusion=self.fusion,
                    is_blender=self.is_blender,
                    is_6dof=self.is_6dof,
                ).cuda()
                tmp_students.append(s)

            warm_init_all_students(
                teacher_network=self.teacher,
                student_networks=tmp_students,
                student_configs=self.student_configs,
                cfg=warm_init_cfg,
                noise_std_per_student=noise_std_per_student,
            )

            # Copy plane parameters from tmp_students → batched_net slices
            with torch.no_grad():
                for lvl_idx, lvl_planes in enumerate(self.batched_net.hexplane.planes):
                    for pidx, param in enumerate(lvl_planes):
                        for k, s in enumerate(tmp_students):
                            param.data[k] = s.hexplane.planes[lvl_idx][pidx].data.squeeze(0)

            del tmp_students
            logger.info("[ClusteredDeform] Warm-init: copied teacher planes into %d student slices (batched).", n)
            return

        # Sequential path: delegate to warm_init_all_students as before
        from utils.warm_init_utils import warm_init_all_students

        warm_init_all_students(
            teacher_network=self.teacher,
            student_networks=list(self.students),
            student_configs=self.student_configs,
            cfg=warm_init_cfg,
            noise_std_per_student=noise_std_per_student,
        )
        logger.info("[ClusteredDeform] Warm-init completed for %d students (sequential).", n)
    
    def get_regularization_loss(
        self, tv_temporal_weight: Optional[float] = None
    ) -> torch.Tensor:
        """Get TV and L1 regularization from all student models."""
        if self.use_batched_students:
            if tv_temporal_weight is not None:
                s_tv, t_tv = self.batched_net.get_plane_tv_loss_split()
                l1 = self.batched_net.get_plane_l1_loss()
                return 1e-3 * s_tv + tv_temporal_weight * t_tv + 1e-4 * l1
            return (
                1e-3 * self.batched_net.get_plane_tv_loss()
                + 1e-4 * self.batched_net.get_plane_l1_loss()
            )
        # Sequential path
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

        For the batched mode this returns a single-element list so the
        parallel-stream code in train.py works unchanged (1 stream, no overhead).
        """
        if self.use_batched_students:
            return [self.get_regularization_loss(tv_temporal_weight=tv_temporal_weight)]
        # Sequential path
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

    # ── Module C: Boundary regularization ────────────────────────────

    def get_boundary_reg_loss(
        self,
        xyz: torch.Tensor,
        cluster_ids: torch.Tensor,
        time_emb: torch.Tensor,
        margin: float = 0.05,
    ) -> torch.Tensor:
        """Penalise deformation magnitude for Gaussians near cluster AABB boundaries.

        For each student *k*, points whose normalised coordinate in any axis
        exceeds ``(1 − margin)`` are considered "near-boundary".  We penalise
        the L2 norm of their predicted ``d_xyz`` to encourage the deformation
        field to smoothly decay toward the AABB edge, preventing grid-sample
        border-clamping artefacts.

        The forward pass is only performed for near-boundary points, keeping
        overhead low when ``margin`` is small.

        Parameters
        ----------
        xyz : Tensor ``(N, 3)``
            Current Gaussian positions (detached).
        cluster_ids : Tensor ``(N,)``
            Per-Gaussian cluster index (−1 = static, skipped).
        time_emb : Tensor ``(N, 1)`` or ``(1, 1)``
            Normalised time stamp.
        margin : float
            Fraction of the normalised range ``[−1, 1]`` considered as the
            boundary zone.  E.g. 0.05 → penalises coords with |x| > 0.95.

        Returns
        -------
        Scalar tensor (zero when no boundary points found).
        """
        device = xyz.device
        total = torch.zeros(1, device=device, dtype=xyz.dtype)

        use_per_point_time = time_emb.shape[0] == xyz.shape[0]

        if self.use_batched_students:
            # Vectorised over K: normalise all cluster points at once
            # aabb_mins/maxs: (K, 3)
            for k in range(self.n_clusters):
                mask = (cluster_ids == k)
                if mask.sum() == 0:
                    continue
                pts_k = xyz[mask]
                aabb_min_k = self.batched_net.aabb_mins[k]
                aabb_max_k = self.batched_net.aabb_maxs[k]
                xyz_norm = (
                    2.0 * (pts_k - aabb_min_k) /
                    (aabb_max_k - aabb_min_k + 1e-8) - 1.0
                )
                near_boundary = (xyz_norm.abs() > (1.0 - margin)).any(dim=-1)
                n_border = int(near_boundary.sum().item())
                if n_border == 0:
                    continue
                pts_border = pts_k[near_boundary]
                t_border = (
                    time_emb[mask][near_boundary] if use_per_point_time
                    else time_emb.expand(n_border, -1)
                )
                # Single-cluster forward through the batched net
                out = self.batched_net(
                    [pts_border], [t_border], [n_border]
                )  # (1, n_border, 10)
                total = total + out[0, :n_border, :3].pow(2).mean()
            return total.squeeze(0)

        # Sequential path
        for k, student in enumerate(self.students):
            mask = (cluster_ids == k)
            if mask.sum() == 0:
                continue

            pts_k = xyz[mask]

            xyz_norm = (
                2.0 * (pts_k - student.aabb_min) /
                (student.aabb_max - student.aabb_min + 1e-8) - 1.0
            )

            near_boundary = (xyz_norm.abs() > (1.0 - margin)).any(dim=-1)
            n_border = int(near_boundary.sum().item())
            if n_border == 0:
                continue

            pts_border = pts_k[near_boundary]
            if use_per_point_time:
                t_border = time_emb[mask][near_boundary]
            else:
                t_border = time_emb.expand(n_border, -1)

            d_xyz_b, _, _ = student(pts_border, t_border)
            total = total + d_xyz_b.pow(2).mean()

        return total.squeeze(0)

    # ── Module D: AABB statistics / TensorBoard visualisation ────────

    def log_cluster_aabb_stats(
        self,
        tb_writer,
        iteration: int,
    ) -> None:
        """Log per-student AABB coverage statistics to TensorBoard.

        For each student *k*, records:

        * ``cluster_aabb/volume_ratio_k{k}`` — teacher AABB volume / student
          AABB volume, i.e. how many times the student's spatial resolution is
          effectively boosted vs. using the global AABB.
        * ``cluster_aabb/extent_x/y/z_k{k}`` — absolute side lengths of the
          student AABB in world units.

        Parameters
        ----------
        tb_writer : SummaryWriter or None
        iteration : int
        """
        if tb_writer is None:
            return

        teacher_vol = (
            (self.teacher.aabb_max - self.teacher.aabb_min).clamp(min=1e-6).prod()
        )

        for k in range(self.n_clusters):
            if self.use_batched_students:
                extent_k = (
                    self.batched_net.aabb_maxs[k] - self.batched_net.aabb_mins[k]
                ).clamp(min=1e-6)
            else:
                extent_k = (
                    self.students[k].aabb_max - self.students[k].aabb_min
                ).clamp(min=1e-6)
            vol_k = extent_k.prod()
            ratio = (teacher_vol / vol_k).item()

            tb_writer.add_scalar(f"cluster_aabb/volume_ratio_k{k}", ratio, iteration)
            tb_writer.add_scalar(f"cluster_aabb/extent_x_k{k}", extent_k[0].item(), iteration)
            tb_writer.add_scalar(f"cluster_aabb/extent_y_k{k}", extent_k[1].item(), iteration)
            tb_writer.add_scalar(f"cluster_aabb/extent_z_k{k}", extent_k[2].item(), iteration)

        logger.info(
            "[ClusteredDeform] AABB stats logged at iter %d "
            "(teacher vol=%.4f)", iteration, teacher_vol.item()
        )
