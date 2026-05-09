"""HexPlane 网格点利用率分析器。

通过 PyTorch forward_pre_hook 追踪 HexPlaneField 中每个平面格点的访问情况，
统计测试集渲染全程中实际被访问的网格点比例（利用率），并输出直方图。

使用方式（render.py 中）：

    from utils.grid_util_tracker import GridUtilizationTracker

    with GridUtilizationTracker(deform, track_teacher=False) as tracker:
        for view in views:
            ...  # 正常渲染，hook 自动统计

    tracker.plot_and_save(save_path, iteration)
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from scene.deform_model import ClusteredDeformModel, DeformModel_4DGS
    from utils.hexplane_utils import HexPlaneField

# (di, dj) 平面对顺序与 HexPlaneField 中 _PLANE_PAIRS 一致
_PLANE_PAIRS: List[Tuple[int, int]] = [
    (0, 1),  # XY  — pidx 0
    (0, 2),  # XZ  — pidx 1
    (0, 3),  # XT  — pidx 2
    (1, 2),  # YZ  — pidx 3
    (1, 3),  # YT  — pidx 4
    (2, 3),  # ZT  — pidx 5
]
_PLANE_NAMES: List[str] = ["XY", "XZ", "XT", "YZ", "YT", "ZT"]
_PLANE_TYPES: List[str] = [
    "spatial", "spatial", "spacetime",
    "spatial", "spacetime", "spacetime",
]
# 空间-时间平面（W 轴为时间）的 pidx 集合
_SPACETIME_PIDX = frozenset({2, 4, 5})

# 颜色：空间平面蓝色，时空平面橙色
_BAR_COLORS: List[str] = [
    "#4c72b0", "#4c72b0", "#dd8452",
    "#4c72b0", "#dd8452", "#dd8452",
]


class GridUtilizationTracker:
    """对 HexPlaneField 的网格访问情况进行非侵入式追踪。

    通过在 HexPlaneField 实例上注册 ``forward_pre_hook``，在渲染过程中
    自动统计每个平面的格点访问情况。完全不修改前向计算逻辑。

    Parameters
    ----------
    deform : ClusteredDeformModel | DeformModel_4DGS
        要追踪的形变模型实例。
    track_teacher : bool
        是否同时追踪 ClusteredDeformModel 的教师场（默认关闭，减少开销）。
    """

    def __init__(self, deform, track_teacher: bool = False) -> None:
        self.deform = deform
        self.track_teacher = track_teacher

        # 追踪数据：entity_key → level → pidx → bool 位图 (H*W,)
        self._used_flat: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = {}
        # 平面尺寸：entity_key → level → pidx → (H, W)
        self._plane_shapes: Dict[str, Dict[int, Dict[int, Tuple[int, int]]]] = {}
        # 已注册的 hook 句柄列表，用于退出时清理
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ── Context manager 接口 ─────────────────────────────────────────

    def __enter__(self) -> "GridUtilizationTracker":
        self._register_hooks()
        return self

    def __exit__(self, *args) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Hook 注册 ────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """根据形变模型类型注册 hook。"""
        # 延迟导入避免循环依赖
        try:
            from scene.deform_model import ClusteredDeformModel, DeformModel_4DGS
        except ImportError:
            ClusteredDeformModel = None
            DeformModel_4DGS = None

        deform = self.deform

        if ClusteredDeformModel is not None and isinstance(deform, ClusteredDeformModel):
            # 学生场（sequential 模式）
            if deform.students is not None:
                for k, student in enumerate(deform.students):
                    key = f"student_{k}"
                    h = student.hexplane.register_forward_pre_hook(
                        self._make_hexplane_hook(key)
                    )
                    self._hooks.append(h)

            # batched 学生场（use_batched_students=True）
            if deform.batched_net is not None:
                # BatchedHexPlaneDeformNetwork 内部的 hexplane 是 BatchedHexPlaneField
                # 退化为单一 "batched_students" 实体统计
                h = deform.batched_net.hexplane.register_forward_pre_hook(
                    self._make_batched_hexplane_hook(deform.n_clusters)
                )
                self._hooks.append(h)

            # 可选：教师场
            if self.track_teacher:
                h = deform.teacher.hexplane.register_forward_pre_hook(
                    self._make_hexplane_hook("teacher")
                )
                self._hooks.append(h)

        elif DeformModel_4DGS is not None and isinstance(deform, DeformModel_4DGS):
            # 4DGS 单场基线
            h = deform.deform.hexplane.register_forward_pre_hook(
                self._make_hexplane_hook("global")
            )
            self._hooks.append(h)
        else:
            # MLP 形变模型（无 HexPlane），跳过
            pass

    # ── Hook 工厂函数 ────────────────────────────────────────────────

    def _make_hexplane_hook(self, entity_key: str):
        """返回一个 forward_pre_hook，追踪 HexPlaneField 的 xyzt 输入访问情况。

        hook 签名：fn(module, args) → None
        其中 args[0] = xyzt: Tensor(N, 4), 已归一化到 [-1, 1]。
        """

        def hook(module: nn.Module, args: tuple) -> None:
            if not args:
                return
            xyzt = args[0].detach()  # (N, 4)
            N = xyzt.shape[0]
            if N == 0:
                return

            # 首次调用时初始化位图（从 module.planes 读取分辨率）
            if entity_key not in self._used_flat:
                self._init_entity_bitmaps(entity_key, module, xyzt.device)

            with torch.no_grad():
                self._mark_visited(entity_key, module, xyzt)

        return hook

    def _make_batched_hexplane_hook(self, n_clusters: int):
        """对 BatchedHexPlaneField 注册 hook，退化为统一追踪（不分 student）。

        BatchedHexPlaneField.forward 接收的 args[0] 形状为 (N, 4)（单个查询批次），
        整体标记为 "batched_students" 实体。
        """

        def hook(module: nn.Module, args: tuple) -> None:
            if not args:
                return
            xyzt = args[0].detach()
            N = xyzt.shape[0]
            if N == 0:
                return

            entity_key = "batched_students"
            if entity_key not in self._used_flat:
                self._init_entity_bitmaps(entity_key, module, xyzt.device)

            with torch.no_grad():
                self._mark_visited(entity_key, module, xyzt)

        return hook

    # ── 位图初始化 ───────────────────────────────────────────────────

    def _init_entity_bitmaps(
        self, entity_key: str, hexplane_module: nn.Module, device: torch.device
    ) -> None:
        """从 HexPlaneField 的 planes 属性读取形状，初始化全零 bool 位图。"""
        self._used_flat[entity_key] = {}
        self._plane_shapes[entity_key] = {}

        planes = hexplane_module.planes  # nn.ModuleList of nn.ParameterList
        num_levels = len(planes)

        for lvl in range(num_levels):
            self._used_flat[entity_key][lvl] = {}
            self._plane_shapes[entity_key][lvl] = {}

            for pidx in range(len(_PLANE_PAIRS)):
                param = planes[lvl][pidx]  # (1, C, H, W) or (K, C, H, W) for batched
                H, W = param.shape[2], param.shape[3]
                self._used_flat[entity_key][lvl][pidx] = torch.zeros(
                    H * W, dtype=torch.bool, device=device
                )
                self._plane_shapes[entity_key][lvl][pidx] = (H, W)

    # ── 访问标记（核心计算）──────────────────────────────────────────

    def _mark_visited(
        self,
        entity_key: str,
        hexplane_module: nn.Module,
        xyzt: torch.Tensor,  # (N, 4)
    ) -> None:
        """将 xyzt 对应的 4 个双线性邻域格点标记为已访问。

        坐标映射（align_corners=True, padding_mode="border"）：
            col_f = (xyzt[:, dj] + 1) / 2 * (W - 1)   → W 轴（列）
            row_f = (xyzt[:, di] + 1) / 2 * (H - 1)   → H 轴（行）
        双线性访问 4 个整数格点：(row0,col0), (row0,col1), (row1,col0), (row1,col1)
        Flat 索引：row * W + col
        """
        num_levels = len(hexplane_module.planes)
        N = xyzt.shape[0]

        # 预先在 GPU 上分配全 1 bool 张量（4N 个，用于 index_put_）
        # 懒惰分配：在循环外分配最大尺寸，避免重复 alloc
        ones_4n = torch.ones(4 * N, dtype=torch.bool, device=xyzt.device)

        for lvl in range(num_levels):
            for pidx, (di, dj) in enumerate(_PLANE_PAIRS):
                H, W = self._plane_shapes[entity_key][lvl][pidx]

                # 归一化坐标 → 像素浮点坐标
                col_f = (xyzt[:, dj] + 1.0) * 0.5 * (W - 1)  # (N,)
                row_f = (xyzt[:, di] + 1.0) * 0.5 * (H - 1)  # (N,)

                # border clamp（对应 padding_mode="border"）
                col_f = col_f.clamp(0.0, float(W - 1))
                row_f = row_f.clamp(0.0, float(H - 1))

                # 4 个邻域整数索引（floor / ceil）
                col0 = col_f.floor().long().clamp(0, W - 1)
                col1 = col_f.ceil().long().clamp(0, W - 1)
                row0 = row_f.floor().long().clamp(0, H - 1)
                row1 = row_f.ceil().long().clamp(0, H - 1)

                # 合并成 flat 索引（4N,），允许重复（index_put_ accumulate=False 无害）
                flat = torch.cat([
                    row0 * W + col0,
                    row0 * W + col1,
                    row1 * W + col0,
                    row1 * W + col1,
                ])  # (4N,)

                # OR 标记到位图（index_put_ 比 scatter_ 更高效）
                self._used_flat[entity_key][lvl][pidx].index_put_(
                    (flat,), ones_4n, accumulate=False
                )

    # ── 统计汇总 ─────────────────────────────────────────────────────

    def compute_stats(self) -> Dict[str, Dict]:
        """计算每个 (entity, level, plane) 的利用率统计。

        Returns
        -------
        dict
            结构为：
            {
                entity_key: {
                    "level":      List[int],
                    "plane":      List[int],
                    "plane_name": List[str],
                    "plane_type": List[str],
                    "util":       List[float],   # [0, 1]
                    "used":       List[int],
                    "total":      List[int],
                    "H":          List[int],
                    "W":          List[int],
                },
                ...
            }
        """
        result: Dict[str, Dict] = {}

        for key, levels in self._used_flat.items():
            rows: Dict[str, list] = {
                "level": [], "plane": [], "plane_name": [],
                "plane_type": [], "util": [], "used": [], "total": [],
                "H": [], "W": [],
            }
            for lvl in sorted(levels.keys()):
                for pidx in sorted(levels[lvl].keys()):
                    bitmap = levels[lvl][pidx]
                    H, W = self._plane_shapes[key][lvl][pidx]
                    total = H * W
                    used = int(bitmap.sum().item())
                    rows["level"].append(lvl)
                    rows["plane"].append(pidx)
                    rows["plane_name"].append(_PLANE_NAMES[pidx])
                    rows["plane_type"].append(_PLANE_TYPES[pidx])
                    rows["util"].append(used / total if total > 0 else 0.0)
                    rows["used"].append(used)
                    rows["total"].append(total)
                    rows["H"].append(H)
                    rows["W"].append(W)
            result[key] = rows

        return result

    # ── 直方图绘制 ───────────────────────────────────────────────────

    def plot_and_save(self, save_path: str, iteration: int) -> None:
        """绘制利用率直方图并保存为 PNG，同时将原始数据写入同路径的 JSON 文件。

        图形布局：
        - 子图行：每个 entity（student_0..K-1 / teacher / global / batched_students）
        - 子图列：每个 level（0, 1, 2, ...）
        - 每个子图：6 个竖条（对应 XY/XZ/XT/YZ/YT/ZT 6 个平面）
          - 蓝色：纯空间平面；橙色：时空平面

        Parameters
        ----------
        save_path : str
            输出 PNG 文件路径。对应 JSON 会保存到同路径（.png → .json）。
        iteration : int
            当前渲染迭代号，用于图题标注。
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # 非交互后端，无需显示器
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            print("[GridUtil] matplotlib 未安装，跳过直方图生成。")
            self._save_json(save_path, iteration)
            return

        stats = self.compute_stats()
        if not stats:
            print("[GridUtil] 没有追踪到任何 HexPlane 访问，跳过绘图。")
            return

        # 按 entity 名称排序：teacher 放最后，其余按字母序
        def _sort_key(k: str) -> str:
            if k == "teacher":
                return "zzz_teacher"
            return k

        entities = sorted(stats.keys(), key=_sort_key)
        n_entities = len(entities)

        # 确定 level 数（所有 entity 取最大值）
        n_levels = max(
            (max(stats[e]["level"]) + 1 if stats[e]["level"] else 0)
            for e in entities
        )
        if n_levels == 0:
            print("[GridUtil] 无有效 level 数据，跳过绘图。")
            return

        fig, axes = plt.subplots(
            n_entities, n_levels,
            figsize=(max(4 * n_levels, 8), max(2.5 * n_entities, 4)),
            squeeze=False,
        )

        for ei, key in enumerate(entities):
            rows = stats[key]
            for lvl in range(n_levels):
                ax = axes[ei][lvl]
                # 筛出该 level 的记录（按 pidx 排序）
                indices = [
                    i for i, l in enumerate(rows["level"]) if l == lvl
                ]
                if not indices:
                    ax.set_visible(False)
                    continue

                util_vals = [rows["util"][i] for i in indices]
                plane_names = [rows["plane_name"][i] for i in indices]
                hw_labels = [
                    f'{rows["H"][i]}×{rows["W"][i]}'
                    for i in indices
                ]
                colors = [_BAR_COLORS[rows["plane"][i]] for i in indices]

                x_pos = list(range(len(indices)))
                bars = ax.bar(x_pos, util_vals, color=colors, alpha=0.85, width=0.6)

                ax.set_xticks(x_pos)
                ax.set_xticklabels(plane_names, fontsize=7)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Util.", fontsize=7)
                ax.yaxis.set_tick_params(labelsize=6)

                # 标题：entity 名 + level + 平面尺寸示例
                _hw_sample = hw_labels[0] if hw_labels else ""
                ax.set_title(
                    f"{key} | L{lvl}\n({_hw_sample})",
                    fontsize=8,
                    pad=2,
                )

                # 在每个 bar 上标注数值
                for bar, v in zip(bars, util_vals):
                    label_y = v + 0.02 if v < 0.95 else v - 0.06
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        label_y,
                        f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6,
                    )

        # 图例
        legend_handles = [
            Patch(color="#4c72b0", label="Spatial (XY/XZ/YZ)"),
            Patch(color="#dd8452", label="Space-time (XT/YT/ZT)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            bbox_to_anchor=(1.0, 1.0),
        )

        model_name = os.path.basename(os.path.dirname(os.path.dirname(save_path)))
        fig.suptitle(
            f"HexPlane Grid Utilization — {model_name} | iter {iteration}",
            fontsize=10,
            y=1.01,
        )
        fig.tight_layout(rect=[0, 0, 0.92, 1.0])

        # 保存 PNG
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[GridUtil] 利用率直方图已保存至: {save_path}")

        # 保存 JSON
        self._save_json(save_path, iteration)

    def _save_json(self, png_path: str, iteration: int) -> None:
        """将 compute_stats() 结果保存为 JSON（供后续分析使用）。"""
        stats = self.compute_stats()
        json_path = png_path.replace(".png", ".json")

        # 将 bool bitmap 的利用率数据序列化（不序列化位图本身）
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[GridUtil] 原始统计数据已保存至: {json_path}")

    def print_summary(self) -> None:
        """在终端打印利用率摘要表（可选，调试用）。"""
        stats = self.compute_stats()
        header = f"{'Entity':<20} {'Level':>5} {'Plane':>6} {'Used':>8} {'Total':>8} {'Util':>6}"
        print("\n" + "=" * len(header))
        print("HexPlane Grid Utilization Summary")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for key in sorted(stats.keys()):
            rows = stats[key]
            for i in range(len(rows["level"])):
                print(
                    f"{key:<20} {rows['level'][i]:>5} "
                    f"{rows['plane_name'][i]:>6} "
                    f"{rows['used'][i]:>8,} {rows['total'][i]:>8,} "
                    f"{rows['util'][i]:>6.1%}"
                )
        print("=" * len(header) + "\n")
