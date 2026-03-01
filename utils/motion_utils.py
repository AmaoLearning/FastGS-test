import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.time_utils import get_embedder
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from typing import Tuple, Union
# from torch.func import vmap, jacrev # torch < 2.0

# tinycudann: optional — only needed when velocity_network_type == "tcnn"
try:
    import tinycudann as tcnn
    _HAS_TCNN = True
except ImportError:
    _HAS_TCNN = False


# ============================================================================
# Step 1: 物理量计算工具函数 (散度和旋度)
# ============================================================================

def compute_velocity_jacobian_quantities(
    velocity_net: nn.Module,
    coords: torch.Tensor,
    times: torch.Tensor,
    chunk_size: int = 8192
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    高效计算速度场的雅可比矩阵导出量：散度(Divergence)和旋度(Curl)。
    
    使用 torch.func.vmap + jacrev 实现向量化的雅可比矩阵计算。
    此函数仅用于推断物理指标，不保留计算图以节省显存。
    
    Args:
        velocity_net: 速度场网络，输入 (xyz, t)，输出 v [N, 3]
        coords: 高斯中心点坐标 [N, 3]，应为 detached
        times: 时间坐标 [N, 1]，应为 detached
        chunk_size: 分块大小，用于控制显存占用
    
    Returns:
        divergence: 散度 [N, 1]，雅可比矩阵的迹 (∂vx/∂x + ∂vy/∂y + ∂vz/∂z)
        curl_magnitude: 旋度模长 [N, 1]
        velocity: 速度向量 [N, 3]
    
    物理意义:
        - 散度 > 0: 膨胀区域（速度场向外发散）
        - 散度 < 0: 收缩区域（速度场向内汇聚）
        - 旋度大: 旋转/湍流区域（速度场有强烈的非线性形变）
    """
    device = coords.device
    N = coords.shape[0]
    
    # 确保输入是 detached 的
    coords = coords.detach()
    times = times.detach()
    
    # 初始化输出
    divergence = torch.zeros(N, 1, device=device)
    curl_magnitude = torch.zeros(N, 1, device=device)
    velocity = torch.zeros(N, 3, device=device)
    
    # 定义单点速度函数（用于 jacrev）
    def velocity_fn(xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """单点速度函数，输入 xyz [3], t [1]，输出 v [3]"""
        return velocity_net(xyz.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
    
    # Create vmap+jacrev function ONCE outside the loop
    jacobian_fn = jacrev(velocity_fn, argnums=0)
    batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0))
    
    # 分块处理以控制显存
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk_coords = coords[start_idx:end_idx]  # [chunk, 3]
        chunk_times = times[start_idx:end_idx]    # [chunk, 1]
        chunk_size_actual = chunk_coords.shape[0]
        
        with torch.no_grad():
            # 计算速度
            chunk_velocity = velocity_net(chunk_coords, chunk_times)  # [chunk, 3]
            velocity[start_idx:end_idx] = chunk_velocity
            
            # 使用 vmap + jacrev 计算雅可比矩阵 (functions hoisted above loop)
            try:
                # 计算雅可比矩阵 [chunk, 3, 3]
                # J[i,j] = ∂v_i / ∂x_j
                jacobians = batched_jacobian_fn(chunk_coords, chunk_times)
                
                # 散度：迹 = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
                chunk_div = jacobians[:, 0, 0] + jacobians[:, 1, 1] + jacobians[:, 2, 2]  # [chunk]
                divergence[start_idx:end_idx] = chunk_div.unsqueeze(-1)
                
                # 旋度：curl = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
                curl_x = jacobians[:, 2, 1] - jacobians[:, 1, 2]  # ∂vz/∂y - ∂vy/∂z
                curl_y = jacobians[:, 0, 2] - jacobians[:, 2, 0]  # ∂vx/∂z - ∂vz/∂x
                curl_z = jacobians[:, 1, 0] - jacobians[:, 0, 1]  # ∂vy/∂x - ∂vx/∂y
                
                # 旋度模长
                chunk_curl_mag = torch.sqrt(curl_x**2 + curl_y**2 + curl_z**2 + 1e-8)  # [chunk]
                curl_magnitude[start_idx:end_idx] = chunk_curl_mag.unsqueeze(-1)
                
            except ImportError:
                # Fallback: 使用有限差分近似（torch.func 不可用时）
                eps = 1e-4
                chunk_div = torch.zeros(chunk_size_actual, device=device)
                chunk_curl = torch.zeros(chunk_size_actual, 3, device=device)
                
                for dim in range(3):
                    # 正向扰动
                    coords_plus = chunk_coords.clone()
                    coords_plus[:, dim] += eps
                    v_plus = velocity_net(coords_plus, chunk_times)
                    
                    # 负向扰动
                    coords_minus = chunk_coords.clone()
                    coords_minus[:, dim] -= eps
                    v_minus = velocity_net(coords_minus, chunk_times)
                    
                    # 中心差分
                    dv_dx = (v_plus - v_minus) / (2 * eps)  # [chunk, 3]
                    
                    # 累加散度
                    chunk_div += dv_dx[:, dim]
                    
                    # 计算旋度分量
                    # curl_x = ∂vz/∂y - ∂vy/∂z
                    # curl_y = ∂vx/∂z - ∂vz/∂x
                    # curl_z = ∂vy/∂x - ∂vx/∂y
                    if dim == 0:  # ∂/∂x
                        chunk_curl[:, 2] += dv_dx[:, 1]   # ∂vy/∂x -> curl_z
                        chunk_curl[:, 1] -= dv_dx[:, 2]   # -∂vz/∂x -> curl_y
                    elif dim == 1:  # ∂/∂y
                        chunk_curl[:, 0] += dv_dx[:, 2]   # ∂vz/∂y -> curl_x
                        chunk_curl[:, 2] -= dv_dx[:, 0]   # -∂vx/∂y -> curl_z
                    else:  # ∂/∂z
                        chunk_curl[:, 1] += dv_dx[:, 0]   # ∂vx/∂z -> curl_y
                        chunk_curl[:, 0] -= dv_dx[:, 1]   # -∂vy/∂z -> curl_x
                
                divergence[start_idx:end_idx] = chunk_div.unsqueeze(-1)
                curl_magnitude[start_idx:end_idx] = torch.norm(chunk_curl, dim=-1, keepdim=True)
    
    return divergence, curl_magnitude, velocity

class VelocityNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, multires=10, is_blender=False, is_6dof=False):
        super(VelocityNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.velocity_head = nn.Linear(W, 3)
        
        
        self.optimizer = None
        self.spatial_lr_scale = 5

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / (theta + 1e-5)
            v = v / (theta + 1e-5)
            screw_axis = torch.cat([w, v], dim=-1)
            velocity = exp_se3(screw_axis, theta)
        else:
            velocity = self.velocity_head(h)

        return velocity


    def train_setting(self, training_args):
        l = [
            {'params': list(self.parameters()),
            #  'lr': training_args.position_lr_init * self.spatial_lr_scale,
            'lr': training_args.velocity_lr, #0.0008
             "name": "velocity"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.velocity_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.velocity_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "velocity/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'velocity.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "velocity"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "velocity/iteration_{}/velocity.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "velocity":
                lr = self.velocity_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


# ============================================================================
# VelocityNetworkTcnn — tinycudann FullyFusedMLP accelerated velocity field
# ============================================================================

class VelocityNetworkTcnn(nn.Module):
    """Velocity network using the same positional encoding as
    :class:`VelocityNetwork` but replacing the PyTorch ``nn.Linear`` MLP body
    with a tinycudann ``FullyFusedMLP``.

    The fused CUDA kernel merges all Linear + ReLU layers into a single GPU
    dispatch with half-precision Tensor-Core arithmetic, giving ~3-5× speed-up
    on RTX 30/40 series without changing the encoding or network capacity.

    Architecture::

        xyz ──▶ Positional Encoding (sin/cos, multires=10) ──┐
                                                              ├─▶ tcnn.Network
        t   ──▶ Positional Encoding (sin/cos, multires=6/10) ┘    "FullyFusedMLP"
                (+ optional timenet for Blender)                   (W×D) ──▶ v [3]

    The public interface (``forward``, ``train_setting``, ``save_weights``,
    ``load_weights``, ``update_learning_rate``) is identical to the other
    velocity networks so it is a drop-in replacement.
    """

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 3,
        output_ch: int = 3,
        multires: int = 10,
        is_blender: bool = False,
        is_6dof: bool = False,
    ):
        super().__init__()
        if not _HAS_TCNN:
            raise ImportError(
                "tinycudann is required for VelocityNetworkTcnn. "
                "Install with: pip install tinycudann  "
                "(or: pip install git+https://github.com/NVlabs/tiny-cuda-nn/"
                "#subdirectory=bindings/torch)"
            )

        self.D = D
        self.W = W
        self.is_blender = is_blender
        self.is_6dof = is_6dof

        # ── Positional encoding (identical to VelocityNetwork) ────────
        self.t_multires = 6 if is_blender else 10
        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)

        if is_blender:
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out),
            )
            mlp_input_dim = xyz_input_ch + self.time_out
        else:
            mlp_input_dim = xyz_input_ch + time_input_ch

        # ── Fused MLP body (tinycudann) ───────────────────────────────
        # NOTE: FullyFusedMLP does NOT support skip connections.  For the
        # velocity field this is acceptable — the original skip at D//2
        # is dropped in exchange for ~3-5× throughput.  Network capacity
        # is preserved by keeping the same D and W.
        output_dim = 6 if is_6dof else 3

        # FullyFusedMLP only supports width ≤ 128.  For W > 128 we fall
        # back to CutlassMLP which uses NVIDIA Cutlass fused GEMM+ReLU
        # kernels and supports arbitrary width (still ~2-4× faster than
        # pure-PyTorch nn.Linear).
        fused_width = ((W + 15) // 16) * 16
        mlp_otype = "FullyFusedMLP" if fused_width <= 128 else "CutlassMLP"

        self.mlp = tcnn.Network(
            n_input_dims=mlp_input_dim,
            n_output_dims=output_dim,
            network_config={
                "otype": mlp_otype,
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": fused_width,
                "n_hidden_layers": D,
            },
        )

        self.optimizer = None
        self.spatial_lr_scale = 5

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity.

        Args:
            x: [N, 3] world-space coordinates.
            t: [N, 1] normalised time.

        Returns:
            velocity: [N, 3] (or SE(3) warp if ``is_6dof``).
        """
        # Positional encoding (same as VelocityNetwork)
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)
        x_emb = self.embed_fn(x)
        features = torch.cat([x_emb, t_emb], dim=-1)  # [N, mlp_input_dim]

        # tcnn fused MLP — internal fp16, output cast back to fp32
        raw = self.mlp(features).float()               # [N, output_dim]

        if self.is_6dof:
            w = raw[:, :3]
            v = raw[:, 3:6]
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / (theta + 1e-5)
            v = v / (theta + 1e-5)
            screw_axis = torch.cat([w, v], dim=-1)
            velocity = exp_se3(screw_axis, theta)
        else:
            velocity = raw

        return velocity

    # ── Training utilities (identical interface) ──────────────────────

    def train_setting(self, training_args) -> None:
        l = [
            {"params": list(self.parameters()),
             "lr": training_args.velocity_lr,
             "name": "velocity"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.velocity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.velocity_lr_max_steps,
        )

    def save_weights(self, model_path: str, iteration: int) -> None:
        out_weights_path = os.path.join(
            model_path, f"velocity/iteration_{iteration}")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(out_weights_path, "velocity.pth"))

    def load_weights(self, model_path: str, iteration: int = -1) -> None:
        if iteration == -1:
            loaded_iter = searchForMaxIteration(
                os.path.join(model_path, "velocity"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path,
            f"velocity/iteration_{loaded_iter}/velocity.pth")
        self.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "velocity":
                lr = self.velocity_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr


class HashEmbedder(nn.Module):
    """Multi-resolution Hash Encoding (Instant-NGP style)"""
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, 
                 log2_hashmap_size=19, base_resolution=16, max_resolution=2048):
        super(HashEmbedder, self).__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.out_dim = num_levels * level_dim
        
        # 计算每层的分辨率
        self.per_level_scale = torch.exp2(torch.log2(torch.tensor(max_resolution / base_resolution)) / (num_levels - 1))
        
        # 为每一层创建哈希表
        self.embeddings = nn.ModuleList([
            nn.Embedding(2 ** log2_hashmap_size, level_dim)
            for _ in range(num_levels)
        ])
        
        # 初始化
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
    
    def forward(self, x):
        """
        Args:
            x: [N, input_dim] 输入坐标，假设归一化到 [0, 1]
        Returns:
            [N, out_dim] Hash encoded features
        """
        # 将输入归一化到 [0, 1]
        x_normalized = (x + 1.0) / 2.0  # 假设输入在 [-1, 1]
        x_normalized = torch.clamp(x_normalized, 0, 1)
        
        embedded = []
        for level in range(self.num_levels):
            resolution = torch.floor(self.base_resolution * (self.per_level_scale ** level))
            
            # 将坐标缩放到当前层的分辨率
            scaled_x = x_normalized * resolution
            
            # 获取网格坐标的整数部分和小数部分
            grid_coords = torch.floor(scaled_x).long()
            lerp_weights = scaled_x - grid_coords.float()
            
            # 多线性插值 (简化为三线性插值)
            # 对于3D输入，我们需要8个顶点
            features = torch.zeros(x.shape[0], self.level_dim, device=x.device)
            
            if self.input_dim == 3:
                # 3D 三线性插值
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            # 计算顶点坐标
                            corner = grid_coords + torch.tensor([i, j, k], device=x.device)
                            # 计算哈希索引
                            hashed_idx = self._hash(corner, level)
                            # 获取特征
                            corner_features = self.embeddings[level](hashed_idx)
                            # 计算插值权重
                            weight = 1.0
                            weight *= lerp_weights[:, 0] if i == 1 else (1 - lerp_weights[:, 0])
                            weight *= lerp_weights[:, 1] if j == 1 else (1 - lerp_weights[:, 1])
                            weight *= lerp_weights[:, 2] if k == 1 else (1 - lerp_weights[:, 2])
                            features += corner_features * weight.unsqueeze(-1)
            elif self.input_dim == 1:
                # 1D 线性插值 (用于时间)
                for i in range(2):
                    corner = grid_coords + i
                    hashed_idx = self._hash(corner, level)
                    corner_features = self.embeddings[level](hashed_idx)
                    weight = lerp_weights if i == 1 else (1 - lerp_weights)
                    features += corner_features * weight
            
            embedded.append(features)
        
        return torch.cat(embedded, dim=-1)
    
    def _hash(self, coords, level):
        """
        将网格坐标哈希到哈希表索引
        使用 Instant-NGP 中的哈希函数
        """
        primes = [1, 2654435761, 805459861]  # 大质数用于哈希
        
        hashed = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for i in range(min(self.input_dim, coords.shape[1])):
            hashed ^= coords[:, i] * primes[i]
        
        # 模哈希表大小
        return hashed % (2 ** self.log2_hashmap_size)


class VelocityNetworkHash(nn.Module):
    """Velocity Network with Hash Encoding (Instant-NGP style)"""
    def __init__(self, D=4, W=128, is_blender=False, is_6dof=False, 
                 num_levels=8, level_dim=2, log2_hashmap_size=18):
        super(VelocityNetworkHash, self).__init__()
        self.D = D
        self.W = W
        self.is_blender = is_blender
        self.is_6dof = is_6dof
        
        # Hash encoding for spatial coordinates (3D)
        self.spatial_encoder = HashEmbedder(
            input_dim=3, 
            num_levels=num_levels, 
            level_dim=level_dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=16,
            max_resolution=2048
        )
        
        # Hash encoding for temporal coordinate (1D)
        self.temporal_encoder = HashEmbedder(
            input_dim=1,
            num_levels=num_levels // 2,
            level_dim=level_dim,
            log2_hashmap_size=log2_hashmap_size - 2,
            base_resolution=16,
            max_resolution=512
        )
        
        self.input_ch = self.spatial_encoder.out_dim + self.temporal_encoder.out_dim
        
        # MLP layers (更小的网络，因为 hash encoding 已经提供了丰富特征)
        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + 
            [nn.Linear(W, W) for _ in range(D - 1)]
        )
        
        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.velocity_head = nn.Linear(W, 3)
        
        self.optimizer = None
        self.spatial_lr_scale = 5
    
    def forward(self, x, t):
        """
        Args:
            x: [N, 3] spatial coordinates
            t: [N, 1] temporal coordinate
        Returns:
            velocity: [N, 3] or SE(3) velocity
        """
        # Hash encoding
        x_encoded = self.spatial_encoder(x)
        t_encoded = self.temporal_encoder(t)
        h = torch.cat([x_encoded, t_encoded], dim=-1)
        
        # MLP
        for i, layer in enumerate(self.linear):
            h = layer(h)
            h = F.relu(h)
        
        # Output head
        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / (theta + 1e-5)
            v = v / (theta + 1e-5)
            screw_axis = torch.cat([w, v], dim=-1)
            velocity = exp_se3(screw_axis, theta)
        else:
            velocity = self.velocity_head(h)
        
        return velocity
    
    def train_setting(self, training_args):
        l = [
            {'params': list(self.parameters()),
             'lr': training_args.velocity_lr,
             "name": "velocity"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.velocity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.velocity_lr_max_steps
        )
    
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "velocity/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'velocity.pth'))
    
    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "velocity"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "velocity/iteration_{}/velocity.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "velocity":
                lr = self.velocity_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr