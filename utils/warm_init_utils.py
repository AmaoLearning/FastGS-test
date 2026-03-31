"""
warm_init_utils.py
==================
教师 HexPlane → 学生 HexPlane 降采样热启动工具。

支持:
  - 平面分辨率双线性降采样 (方案 A)
  - 特征维度截断/PCA 压缩 (方案 B)
  - MLP 权重子矩阵迁移 (方案 C)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Literal, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

# 延迟导入以避免循环依赖
if TYPE_CHECKING:
    from utils.hexplane_utils import HexPlaneField

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 配置数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WarmInitConfig:
    """热启动的全局配置，可通过命令行参数注入。"""

    # ── 总开关 ──
    enabled: bool = True
    """是否启用热启动。False 时函数为空操作。"""

    # ── 方案 A:分辨率降采样 ──
    downsample_planes: bool = True
    """是否对平面做分辨率降采样。"""

    interpolation_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear"
    """插值模式，bilinear 速度快且效果好。"""

    align_corners: bool = False
    """F.interpolate 的 align_corners 参数。"""

    # ── 方案 B:特征维度压缩 ──
    compress_feat_dim: bool = True
    """是否压缩特征通道维度。"""

    feat_compression_method: Literal["truncate", "pca", "random_proj"] = "truncate"
    """特征压缩方法：truncate(截断) / pca(主成分) / random_proj(随机投影)。"""

    # ── 方案 C:MLP 权重迁移 ──
    transfer_mlp: bool = True
    """是否迁移 MLP 权重。"""

    # ── 幅值归一化 ──
    normalize_scale: bool = True
    """迁移后是否对平面幅值做归一化，保持 L2 范数一致。"""

    # ── 按簇裁剪 ──
    cluster_aware: bool = False
    """
    是否在将教师平面迁移给学生前，先按该簇的空间范围裁剪。
    当前版本预留接口，暂未实现。
    """

    # ── 噪声 ──
    noise_std: float = 1e-4
    """
    迁移后添加高斯噪声，用于打破多个学生之间的对称性。
    默认 1e-4，0.0 表示不添加噪声。
    """


# ──────────────────────────────────────────────────────────────────────────────
# 平面级别映射
# ──────────────────────────────────────────────────────────────────────────────

def _build_level_mapping(
    teacher_n_levels: int,
    student_n_levels: int,
) -> List[int]:
    """
    建立学生级别 → 教师级别的映射列表。

    策略:
      - 若学生级别数 ≤ 教师，取前 student_n_levels 个教师级别。
      - 若学生级别数 > 教师，超出部分重用教师最高级别。

    Returns:
        mapping[i] = 教师级别索引，表示学生 level i 应从哪个教师 level 初始化。
    """
    mapping = []
    for s_lvl in range(student_n_levels):
        t_lvl = min(s_lvl, teacher_n_levels - 1)
        mapping.append(t_lvl)
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# 特征维度压缩
# ──────────────────────────────────────────────────────────────────────────────

def _compress_feat_dim(
    plane: torch.Tensor,
    target_dim: int,
    method: str = "truncate",
) -> torch.Tensor:
    """
    将平面 [1, C_src, H, W] 的特征维度压缩到 [1, C_tgt, H, W]。

    Args:
        plane: 教师平面张量，形状 [1, C_src, H, W]。
        target_dim: 目标特征维度 C_tgt(≤ C_src)。
        method: "truncate" / "pca" / "random_proj"。

    Returns:
        压缩后的张量，形状 [1, C_tgt, H, W]。
    """
    _, C_src, H, W = plane.shape
    assert target_dim <= C_src, (
        f"目标维度 {target_dim} 大于源维度 {C_src}，无需压缩。"
    )

    if target_dim == C_src:
        return plane.clone()

    if method == "truncate":
        return plane[:, :target_dim, :, :].clone()

    elif method == "pca":
        # reshape 为 [H*W, C_src]
        feat = plane.squeeze(0).permute(1, 2, 0).reshape(-1, C_src)  # [N, C]
        feat_f = feat.float()
        # 中心化
        mean = feat_f.mean(dim=0, keepdim=True)
        feat_centered = feat_f - mean
        # SVD
        try:
            U, S, Vh = torch.linalg.svd(feat_centered, full_matrices=False)
            # 取前 target_dim 个主成分
            result = U[:, :target_dim] * S[:target_dim]   # [N, target_dim]
        except Exception as e:
            logger.warning(f"PCA SVD 失败，回退到截断方案：{e}")
            result = feat_f[:, :target_dim]

        # reshape 回 [1, target_dim, H, W]
        result = result.reshape(H, W, target_dim).permute(2, 0, 1).unsqueeze(0)
        return result.to(plane.dtype)

    elif method == "random_proj":
        # 随机正交投影矩阵 [C_src, target_dim]
        generator = torch.Generator(device=plane.device)
        generator.manual_seed(42)  # 固定种子保证可复现
        R = torch.randn(C_src, target_dim, device=plane.device,
                        dtype=plane.dtype, generator=generator)
        # QR 分解得正交矩阵
        Q, _ = torch.linalg.qr(R)  # [C_src, target_dim]

        feat = plane.squeeze(0).permute(1, 2, 0).reshape(-1, C_src)  # [N, C_src]
        proj = feat @ Q  # [N, target_dim]
        result = proj.reshape(H, W, target_dim).permute(2, 0, 1).unsqueeze(0)
        return result.to(plane.dtype)

    else:
        raise ValueError(f"未知的特征压缩方法：{method}")


# ──────────────────────────────────────────────────────────────────────────────
# 单个平面迁移
# ──────────────────────────────────────────────────────────────────────────────

def _transfer_single_plane(
    teacher_plane: torch.Tensor,
    target_h: int,
    target_w: int,
    target_feat_dim: int,
    cfg: WarmInitConfig,
) -> torch.Tensor:
    """
    将单个教师平面迁移并转换为学生目标形状。

    处理顺序:
      1. 特征维度压缩 (先做，减少后续插值计算量)
      2. 空间分辨率插值
      3. 幅值归一化

    Args:
        teacher_plane: [1, C_t, H_t, W_t]
        target_h, target_w: 学生目标分辨率
        target_feat_dim: 学生目标特征维度
        cfg: 热启动配置

    Returns:
        [1, target_feat_dim, target_h, target_w]
    """
    plane = teacher_plane  # [1, C_t, H_t, W_t]

    # ── 步骤 1: 特征维度压缩 ──
    if cfg.compress_feat_dim and plane.shape[1] != target_feat_dim:
        if plane.shape[1] > target_feat_dim:
            plane = _compress_feat_dim(plane, target_feat_dim,
                                       method=cfg.feat_compression_method)
        elif plane.shape[1] < target_feat_dim:
            # 目标维度更大：用零填充 (不应出现，但防御性处理)
            pad_c = target_feat_dim - plane.shape[1]
            padding = torch.zeros(1, pad_c, plane.shape[2], plane.shape[3],
                                  device=plane.device, dtype=plane.dtype)
            plane = torch.cat([plane, padding], dim=1)
    elif plane.shape[1] != target_feat_dim:
        # 如果未启用压缩但维度不匹配，需要处理
        if plane.shape[1] > target_feat_dim:
            # 截断到目标维度
            plane = plane[:, :target_feat_dim, :, :]
        else:
            # 填充到目标维度
            pad_c = target_feat_dim - plane.shape[1]
            padding = torch.zeros(1, pad_c, plane.shape[2], plane.shape[3],
                                  device=plane.device, dtype=plane.dtype)
            plane = torch.cat([plane, padding], dim=1)

    # ── 步骤 2: 空间/时间分辨率插值 ──
    if cfg.downsample_planes:
        curr_h, curr_w = plane.shape[2], plane.shape[3]
        if (curr_h, curr_w) != (target_h, target_w):
            plane = F.interpolate(
                plane,
                size=(target_h, target_w),
                mode=cfg.interpolation_mode,
                align_corners=cfg.align_corners if cfg.interpolation_mode != "nearest" else None,
            )

    # ── 步骤 3: 幅值归一化 ──
    if cfg.normalize_scale:
        src_norm = teacher_plane.norm(dim=1, keepdim=True).mean().item()
        tgt_norm = plane.norm(dim=1, keepdim=True).mean().item()
        if tgt_norm > 1e-8:
            plane = plane * (src_norm / tgt_norm)

    return plane.detach().clone()


# ──────────────────────────────────────────────────────────────────────────────
# HexPlane 六平面迁移 (一个分辨率级别)
# ──────────────────────────────────────────────────────────────────────────────

def transfer_hexplane_level(
    teacher_planes_at_level: List[torch.Tensor],
    student_spatial_res: int,
    student_time_res: int,
    student_feat_dim: int,
    cfg: WarmInitConfig,
) -> List[torch.Tensor]:
    """
    将教师某个分辨率级别的六个平面迁移到学生对应级别。

    平面索引约定 (与 FastGS HexPlaneField 一致):
      0: XY  [1, C, S_y,  S_x ]
      1: XZ  [1, C, S_z,  S_x ]
      2: YZ  [1, C, S_z,  S_y ]
      3: XT  [1, C, T,    S_x ]
      4: YT  [1, C, T,    S_y ]
      5: ZT  [1, C, T,    S_z ]

    Args:
        teacher_planes_at_level: 长度为 6 的列表，每项形状见上。
        student_spatial_res: 学生空间分辨率 (S_x = S_y = S_z)。
        student_time_res: 学生时间分辨率 (T)。
        student_feat_dim: 学生特征维度 C。
        cfg: 热启动配置。

    Returns:
        长度为 6 的列表，每项已转换为学生目标形状。
    """
    S = student_spatial_res
    T = student_time_res

    # 目标尺寸：每个平面的 (H, W)
    target_hw = [
        (S, S),  # XY
        (S, S),  # XZ
        (S, S),  # YZ
        (T, S),  # XT
        (T, S),  # YT
        (T, S),  # ZT
    ]

    result = []
    for i, (t_plane, (th, tw)) in enumerate(zip(teacher_planes_at_level, target_hw)):
        transferred = _transfer_single_plane(t_plane, th, tw, student_feat_dim, cfg)
        result.append(transferred)
        logger.debug(
            f"  平面 {i}: 教师 {tuple(t_plane.shape)} → 学生 {tuple(transferred.shape)}"
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# MLP 权重迁移
# ──────────────────────────────────────────────────────────────────────────────

def transfer_mlp_weights(
    teacher_mlp: nn.Module,
    student_mlp: nn.Module,
    cfg: WarmInitConfig,
) -> None:
    """
    将教师 MLP 的权重截断迁移到学生 MLP(原地修改 student_mlp)。

    对每一个参数张量，按学生维度截取教师子矩阵。若某维度学生比教师大，
    则保留学生原始初始化 (避免零填充带来的梯度消失)。

    Args:
        teacher_mlp: 教师 MLP(nn.Module，含若干 nn.Linear / nn.LayerNorm)。
        student_mlp: 学生 MLP(原地修改)。
        cfg: 热启动配置。
    """
    if not cfg.transfer_mlp:
        return

    t_state = teacher_mlp.state_dict()
    s_state = student_mlp.state_dict()

    new_state = {}
    for key, s_param in s_state.items():
        if key not in t_state:
            logger.debug(f"  MLP 参数 '{key}' 在教师中不存在，保留学生初始化。")
            new_state[key] = s_param
            continue

        t_param = t_state[key]
        if t_param.shape == s_param.shape:
            new_state[key] = t_param.detach().clone()
            continue

        # 按各维度截取
        slices = tuple(
            slice(0, min(s, t)) for s, t in zip(s_param.shape, t_param.shape)
        )
        new_val = s_param.clone()
        new_val[slices] = t_param[slices].detach()
        new_state[key] = new_val
        logger.debug(
            f"  MLP 参数 '{key}': 教师 {tuple(t_param.shape)} → 学生 {tuple(s_param.shape)}"
        )

    student_mlp.load_state_dict(new_state)


# ──────────────────────────────────────────────────────────────────────────────
# 顶层接口：完整的学生 HexPlane 热启动
# ──────────────────────────────────────────────────────────────────────────────

def warm_init_student_from_teacher(
    teacher_network: nn.Module,
    student_network: nn.Module,
    student_spatial_resolutions: List[int],
    student_time_resolutions: List[int],
    student_feat_dim: int,
    cfg: WarmInitConfig,
    noise_std: Optional[float] = None,
) -> None:
    """
    将教师 HexPlaneDeformNetwork 的参数降采样迁移到学生网络 (原地修改)。

    此函数假设 teacher_network 和 student_network 都遵循 FastGS 的
    HexPlaneDeformNetwork 接口，即:
      - network.grids: nn.ParameterList，按
        [level_0_plane_0, ..., level_0_plane_5,
         level_1_plane_0, ..., level_1_plane_5, ...] 排列
      - network.decoder: nn.Sequential(位移解码器)

    Args:
        teacher_network: 已收敛的教师网络 (参数冻结)。
        student_network: 待初始化的学生网络 (原地修改)。
        student_spatial_resolutions: 学生空间分辨率列表，如 [64, 128]。
        student_time_resolutions: 学生时间分辨率列表，如 [64, 128]。
        student_feat_dim: 学生特征维度。
        cfg: 热启动配置。
        noise_std: 可选，添加到迁移平面的高斯噪声标准差。
    """
    if not cfg.enabled:
        logger.info("WarmInit 已禁用，跳过热启动。")
        return

    logger.info(">>> 开始 HexPlane 热启动初始化 >>>")

    # ── 读取教师平面参数 ──
    teacher_grids = _extract_grids(teacher_network)
    teacher_n_levels = len(teacher_grids)
    student_n_levels = len(student_spatial_resolutions)

    logger.info(
        f"  教师级别数：{teacher_n_levels}, 学生级别数：{student_n_levels}"
    )

    # ── 建立级别映射 ──
    level_mapping = _build_level_mapping(teacher_n_levels, student_n_levels)

    # ── 逐级别迁移平面 ──
    new_planes: List[torch.Tensor] = []
    for s_lvl, t_lvl in enumerate(level_mapping):
        t_planes = teacher_grids[t_lvl]  # 6 个教师平面
        s_planes = transfer_hexplane_level(
            teacher_planes_at_level=t_planes,
            student_spatial_res=student_spatial_resolutions[s_lvl],
            student_time_res=student_time_resolutions[s_lvl],
            student_feat_dim=student_feat_dim,
            cfg=cfg,
        )
        new_planes.extend(s_planes)
        logger.info(
            f"  级别 {s_lvl} (源自教师级别 {t_lvl}): "
            f"空间 {student_spatial_resolutions[s_lvl]}, "
            f"时间 {student_time_resolutions[s_lvl]}, "
            f"feat_dim {student_feat_dim}"
        )

    # ── 将迁移后的平面写入学生网络 ──
    _inject_grids(student_network, new_planes, noise_std=noise_std)

    # ── MLP 权重迁移 ──
    if cfg.transfer_mlp:
        teacher_mlp = _get_mlp(teacher_network)
        student_mlp = _get_mlp(student_network)
        if teacher_mlp is not None and student_mlp is not None:
            transfer_mlp_weights(teacher_mlp, student_mlp, cfg)
            logger.info("  MLP 权重迁移完成。")
        else:
            logger.warning("  未找到 MLP 模块，跳过 MLP 迁移。")

    logger.info(">>> HexPlane 热启动初始化完成 >>>")


# ──────────────────────────────────────────────────────────────────────────────
# 内部辅助函数：适配 FastGS HexPlaneDeformNetwork 接口
# ──────────────────────────────────────────────────────────────────────────────

def _extract_grids(network: nn.Module) -> List[List[torch.Tensor]]:
    """
    从 HexPlaneDeformNetwork 中提取平面参数，返回按级别组织的列表。

    FastGS 的 HexPlaneDeformNetwork 包含 HexPlaneField 模块，其 planes 属性为:
      [lvl0_plane0, lvl0_plane1, ..., lvl0_plane5,
       lvl1_plane0, ..., lvl1_plane5, ...]

    每 6 个为一组 (XY, XZ, YZ, XT, YT, ZT)。
    """
    grids = []

    # 尝试直接访问 hexplane 子模块 (HexPlaneDeformNetwork 结构)
    hexplane_module = None
    if hasattr(network, 'hexplane'):
        hexplane_module = network.hexplane
    else:
        # 若网络包含子模块 HexPlaneField，递归查找
        for name, module in network.named_modules():
            if hasattr(module, 'planes'):
                hexplane_module = module
                break

    if hexplane_module is None:
        # 尝试其他常见字段名
        param_list = None
        for attr_name in ["grids", "planes", "hex_planes", "plane_list"]:
            if hasattr(network, attr_name):
                param_list = getattr(network, attr_name)
                break
        
        if param_list is None:
            raise AttributeError(
                "无法在 network 中找到 HexPlane 平面参数。"
                "请检查字段名并修改 _extract_grids()。"
            )
        
        # 按 6 个一组分级别
        all_planes = list(param_list)
        assert len(all_planes) % 6 == 0, (
            f"平面数量 {len(all_planes)} 不是 6 的倍数，请检查网络结构。"
        )
        n_levels = len(all_planes) // 6
        for lvl in range(n_levels):
            grids.append(all_planes[lvl * 6: lvl * 6 + 6])
        return grids

    # 从 HexPlaneField 中提取 planes (ModuleList of ParameterList)
    all_planes = []
    for lvl_planes in hexplane_module.planes:
        all_planes.extend(list(lvl_planes))

    assert len(all_planes) % 6 == 0, (
        f"平面数量 {len(all_planes)} 不是 6 的倍数，请检查网络结构。"
    )
    n_levels = len(all_planes) // 6
    for lvl in range(n_levels):
        grids.append(all_planes[lvl * 6: lvl * 6 + 6])

    return grids


def _inject_grids(
    network: nn.Module,
    new_planes: List[torch.Tensor],
    noise_std: Optional[float] = None,
) -> None:
    """
    将 new_planes 写入 student_network 的平面参数 (原地)。

    Args:
        network: 学生网络。
        new_planes: 按 [lvl0_p0, ..., lvl0_p5, lvl1_p0, ...] 排列的平面列表。
        noise_std: 可选噪声标准差。
    """
    # 尝试直接访问 hexplane 子模块 (HexPlaneDeformNetwork 结构)
    hexplane_module = None
    if hasattr(network, 'hexplane'):
        hexplane_module = network.hexplane
    else:
        # 若网络包含子模块 HexPlaneField，递归查找
        for name, module in network.named_modules():
            if hasattr(module, 'planes'):
                hexplane_module = module
                break

    if hexplane_module is None:
        # 尝试其他常见字段名
        param_list = None
        for attr_name in ["grids", "planes", "hex_planes", "plane_list"]:
            if hasattr(network, attr_name):
                param_list = getattr(network, attr_name)
                break
        
        if param_list is None:
            raise AttributeError("无法找到学生网络的平面参数列表。")
        
        assert len(param_list) == len(new_planes), (
            f"平面数量不匹配：学生有 {len(param_list)} 个，"
            f"但迁移了 {len(new_planes)} 个。"
        )
        
        with torch.no_grad():
            for param, new_val in zip(param_list, new_planes):
                assert param.shape == new_val.shape, (
                    f"形状不匹配：param {param.shape} vs new_val {new_val.shape}"
                )
                noise = 0.0
                if noise_std is not None and noise_std > 0.0:
                    noise = torch.randn_like(new_val) * noise_std
                param.copy_(new_val + noise)
        return

    # 从 HexPlaneField 中获取 planes (ModuleList of ParameterList)
    all_planes = []
    for lvl_planes in hexplane_module.planes:
        all_planes.extend(list(lvl_planes))

    assert len(all_planes) == len(new_planes), (
        f"平面数量不匹配：学生有 {len(all_planes)} 个，"
        f"但迁移了 {len(new_planes)} 个。"
    )

    with torch.no_grad():
        for param, new_val in zip(all_planes, new_planes):
            assert param.shape == new_val.shape, (
                f"形状不匹配：param {param.shape} vs new_val {new_val.shape}"
            )
            noise = 0.0
            if noise_std is not None and noise_std > 0.0:
                noise = torch.randn_like(new_val) * noise_std
            param.copy_(new_val + noise)


def _get_mlp(network: nn.Module) -> Optional[nn.Module]:
    """获取网络中的 MLP 解码器模块。"""
    for attr_name in ["decoder", "mlp", "deform_mlp", "net"]:
        if hasattr(network, attr_name):
            return getattr(network, attr_name)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 批量初始化接口 (ClusteredDeformModel 使用)
# ──────────────────────────────────────────────────────────────────────────────

def warm_init_all_students(
    teacher_network: nn.Module,
    student_networks: List[nn.Module],
    student_configs: List[dict],
    cfg: WarmInitConfig,
    noise_std_per_student: Optional[List[float]] = None,
) -> None:
    """
    对所有学生网络执行热启动初始化。

    Args:
        teacher_network: 教师网络 (参数已冻结)。
        student_networks: 学生网络列表 (与 ClusteredDeformModel.students 对应)。
        student_configs: 每个学生的容量配置，格式:
            [{"spatial_resolutions": [64, 128],
              "time_resolutions": [64, 128],
              "feat_dim": 8}, ...]
        cfg: 热启动配置。
        noise_std_per_student: 可选，每个学生的噪声标准差 (打破对称性)。
            若为 None，使用 cfg.noise_std。
    """
    if not cfg.enabled:
        return

    if noise_std_per_student is None:
        noise_std_per_student = [cfg.noise_std] * len(student_networks)

    for i, (student, s_cfg, noise_std) in enumerate(
        zip(student_networks, student_configs, noise_std_per_student)
    ):
        logger.info(f"\n== 热启动学生 {i} (簇 {i}) ==")
        warm_init_student_from_teacher(
            teacher_network=teacher_network,
            student_network=student,
            student_spatial_resolutions=s_cfg["spatial_resolutions"],
            student_time_resolutions=s_cfg.get("time_resolutions",
                                                s_cfg["spatial_resolutions"]),
            student_feat_dim=s_cfg["feat_dim"],
            cfg=cfg,
            noise_std=noise_std,
        )
