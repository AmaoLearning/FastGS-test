"""
test_warm_init.py
=================
HexPlane 热启动初始化的快速测试脚本。

运行：
    python scripts/test_warm_init.py
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from utils.hexplane_utils import HexPlaneDeformNetwork
from utils.warm_init_utils import (
    WarmInitConfig,
    warm_init_student_from_teacher,
    _extract_grids,
)


def test_warm_init_basic():
    """测试基本热启动功能。"""
    print("\n=== 测试 1: 基本热启动 ===")
    
    # 创建教师网络 (3 级分辨率，feat_dim=16)
    teacher = HexPlaneDeformNetwork(
        spatial_resolutions=(64, 128, 256),
        time_resolutions=(64, 128, 256),
        feat_dim=16,
        mlp_hidden_dim=128,
        mlp_num_hidden=2,
        is_blender=True,
    ).cuda()
    
    # 创建学生网络 (2 级分辨率，feat_dim=8)
    student = HexPlaneDeformNetwork(
        spatial_resolutions=(64, 128),
        time_resolutions=(64, 128),
        feat_dim=8,
        mlp_hidden_dim=64,
        mlp_num_hidden=2,
        is_blender=True,
    ).cuda()
    
    # 保存学生初始参数
    initial_params = {
        name: param.clone()
        for name, param in student.named_parameters()
    }
    
    # 热启动配置
    cfg = WarmInitConfig(
        enabled=True,
        downsample_planes=True,  # 保持开启，避免尺寸不匹配
        feat_compression_method="truncate",
        transfer_mlp=True,  # 仅测试 MLP 迁移
        normalize_scale=True,
        noise_std=1e-4,
    )
    
    # 提取教师网格 (验证用)
    teacher_grids = _extract_grids(teacher)
    print(f"教师网格级别数：{len(teacher_grids)}")
    for i, grids in enumerate(teacher_grids):
        print(f"  级别 {i}: {[g.shape for g in grids]}")
    
    # 冻结教师参数 (模拟实际训练场景)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    
    # 执行热启动
    warm_init_student_from_teacher(
        teacher_network=teacher,
        student_network=student,
        student_spatial_resolutions=[64, 128],
        student_time_resolutions=[64, 128],
        student_feat_dim=8,
        cfg=cfg,
    )
    
    # 验证参数已改变
    changed = False
    for name, param in student.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            changed = True
            break
    
    assert changed, "热启动后学生参数未发生变化！"
    print("✓ 学生参数已更新")
    
    # 验证形状不变
    for name, param in student.named_parameters():
        assert param.shape == initial_params[name].shape, (
            f"参数 '{name}' 形状发生改变：{initial_params[name].shape} → {param.shape}"
        )
    print("✓ 所有参数形状保持不变")
    
    # 验证教师参数未改变
    for param in teacher.parameters():
        assert param.requires_grad == False, "教师参数应被冻结"
    print("✓ 教师参数保持冻结")
    
    print("=== 测试 1 通过 ===\n")


def test_plane_extraction():
    """测试平面参数提取功能。"""
    print("\n=== 测试 2: 平面参数提取 ===")
    
    # 创建网络
    network = HexPlaneDeformNetwork(
        spatial_resolutions=(64, 128),
        time_resolutions=(64, 128),
        feat_dim=16,
        is_blender=True,
    ).cuda()
    
    # 提取网格
    grids = _extract_grids(network)
    
    print(f"提取到 {len(grids)} 个级别")
    for i, level_grids in enumerate(grids):
        print(f"  级别 {i}: {len(level_grids)} 个平面")
        for j, plane in enumerate(level_grids):
            print(f"    平面 {j}: {plane.shape}")
    
    # 验证：应该是 2 级别 × 6 平面
    assert len(grids) == 2, "应该有 2 个级别"
    assert len(grids[0]) == 6, "每级应该有 6 个平面"
    
    # 验证平面形状
    # 级别 0: spatial=64, time=64
    assert grids[0][0].shape == (1, 16, 64, 64), "XY 平面形状错误"  # XY
    assert grids[0][3].shape == (1, 16, 64, 64), "XT 平面形状错误"  # XT
    
    # 级别 1: spatial=128, time=128
    assert grids[1][0].shape == (1, 16, 128, 128), "级别 1 XY 平面形状错误"
    assert grids[1][3].shape == (1, 16, 128, 128), "级别 1 XT 平面形状错误"
    
    print("✓ 平面参数提取验证通过")
    print("=== 测试 2 通过 ===\n")


def test_disabled_warm_init():
    """测试禁用热启动的情况。"""
    print("\n=== 测试 3: 禁用热启动 ===")
    
    teacher = HexPlaneDeformNetwork(
        spatial_resolutions=(64, 128, 256),
        time_resolutions=(64, 128, 256),
        feat_dim=16,
        is_blender=True,
    ).cuda()
    
    student = HexPlaneDeformNetwork(
        spatial_resolutions=(64, 128),
        time_resolutions=(64, 128),
        feat_dim=8,
        is_blender=True,
    ).cuda()
    
    initial_params = {
        name: param.clone()
        for name, param in student.named_parameters()
    }
    
    # 禁用热启动
    cfg = WarmInitConfig(enabled=False)
    
    warm_init_student_from_teacher(
        teacher_network=teacher,
        student_network=student,
        student_spatial_resolutions=[64, 128],
        student_time_resolutions=[64, 128],
        student_feat_dim=8,
        cfg=cfg,
    )
    
    # 验证参数未改变
    for name, param in student.named_parameters():
        torch.testing.assert_close(param, initial_params[name])
    
    print("✓ 禁用热启动时参数保持不变")
    print("=== 测试 3 通过 ===\n")


def test_mlp_transfer():
    """测试 MLP 权重迁移。"""
    print("\n=== 测试 4: MLP 权重迁移 ===")
    
    teacher = HexPlaneDeformNetwork(
        spatial_resolutions=(64,),
        time_resolutions=(64,),
        feat_dim=16,
        mlp_hidden_dim=128,
        is_blender=True,
    ).cuda()
    
    student = HexPlaneDeformNetwork(
        spatial_resolutions=(64,),
        time_resolutions=(64,),
        feat_dim=8,
        mlp_hidden_dim=64,
        is_blender=True,
    ).cuda()
    
    # 保存学生 MLP 初始权重
    initial_mlp_state = {
        name: param.clone()
        for name, param in student.decoder.named_parameters()
    }
    
    cfg = WarmInitConfig(
        enabled=True,
        downsample_planes=False,  # 关闭平面迁移
        feat_compression_method="none",
        transfer_mlp=True,  # 仅测试 MLP 迁移
        noise_std=0.0,
    )
    
    warm_init_student_from_teacher(
        teacher_network=teacher,
        student_network=student,
        student_spatial_resolutions=[64],
        student_time_resolutions=[64],
        student_feat_dim=8,
        cfg=cfg,
    )
    
    # 验证 MLP 权重已改变
    changed = False
    for name, param in student.decoder.named_parameters():
        if not torch.allclose(param, initial_mlp_state[name]):
            changed = True
            break
    
    assert changed, "MLP 权重未发生变化！"
    print("✓ MLP 权重已迁移")
    
    # 验证形状不变
    for name, param in student.decoder.named_parameters():
        assert param.shape == initial_mlp_state[name].shape, (
            f"MLP 参数 '{name}' 形状发生改变"
        )
    print("✓ MLP 参数形状保持不变")
    print("=== 测试 4 通过 ===\n")


if __name__ == "__main__":
    print("HexPlane 热启动初始化测试")
    print("=" * 50)
    
    test_plane_extraction()
    test_warm_init_basic()
    test_disabled_warm_init()
    test_mlp_transfer()
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50 + "\n")
