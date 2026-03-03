#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False

        # velocity
        self.use_velocity = False
        self.velocity_network_type = "mlp"  # "mlp", "hash", or "tcnn" - 使用 MLP、纯PyTorch Hash 或 tinycudann 加速

        # optical flow loss
        self.use_flow_loss = False  # 是否启用投影光流损失
        self.use_flow_tv_loss = False  # 是否对光流预测加 TV 正则
        self.use_velocity_smooth = False  # 是否启用时间平滑正则化

        # dynamic mask (用于选择性计算 deform)
        self.use_dynamic_mask = False  # 是否启用动态掩码

        # physics-driven densification (物理驱动致密化)
        self.use_physics_densify = False  # 是否启用物理驱动致密化
        self.use_div_mask = False  # 是否使用散度掩码（控制 Clone），需配合 use_physics_densify 使用，store_true
        self.use_curl_mask = False  # 是否使用旋度掩码（控制 Split），需配合 use_physics_densify 使用，store_true
        # flow_dir 已废弃：光流数据现在由数据集读取器从 optical_flow/ 目录自动加载

        # lazy loading — N3V large-scale dataset (zero OOM)
        self.lazy_load = False
        self.num_images = 300  # number of frames per camera for N3V dataset
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser, sentinel: bool = False):
        self.iterations = 40_000
        self.warm_up = 3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0007
        self.final_prune_from_iter = 15_000
        self.final_prune_until_iter = 30_000
        self.final_prune_interval = 3000

        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0006 
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5

        # velocity
        self.velocity_lr = 0.01  # 降低初始学习率，0.2过高导致不稳定
        self.velocity_lr_max_steps = 40_000
        self.lambda_velocity = 10  # 降低权重，100过高会压制渲染损失
        self.velocity_interval = 5  # 更频繁更新，从10改为5
        self.velocity_loss_thresh = 0.00003  # velocity loss 阈值
        self.velocity_loss_percentile = 30  # 自适应阈值百分比，-1表示使用固定阈值，0-100表示使用自适应阈值（如50表示取中位数作为阈值）
        self.detach_velocity_loss_from_deform = True  # 是否阻断 velocity loss 对形变场的梯度传播
        self.velocity_grad_clip = 1.0  # 速度场梯度裁剪阈值，0表示不裁剪
        
        # velocity temporal smoothness
        self.lambda_velocity_smooth = 0.1  # 时间平滑正则化权重
        self.velocity_smooth_dt = 0.1  # 时间平滑采样的时间间隔（相对于单帧间隔的比例）
        
        # optical flow loss（投影光流监督损失）
        self.lambda_flow = 0.1  # 光流损失权重
        self.flow_loss_from_iter = 6000  # 从第几个 iteration 开始使用光流损失
        self.flow_loss_interval = 5  # 每隔几个 iteration 计算一次光流损失（节省算力）
        self.flow_tv_weight = 0.01  # TV 正则权重
        self.detach_flow_geometry = True  # 是否阻断光流损失对几何（位置/协方差）的梯度传播
        
        # 自适应权重平衡（velocity_loss 和 flow_loss）
        self.adaptive_velocity_weight = False  # 是否启用自适应权重平衡
        self.velocity_flow_target_ratio = 1.0  # 目标损失比例 velocity_loss / flow_loss
        self.adaptive_weight_ema = 0.99  # EMA衰减系数，用于平滑损失历史

        self.dynamic_decay = 0.9  # Leaky Max 的衰减系数；从0.95降低到0.9，更快响应新的速度变化
        self.dynamic_thresh = 0.0001  # 动态阈值，降低阈值让更多高斯被认为是动态的
        self.dynamic_thresh_percentile = 50  # 自适应阈值百分比，从75降低到50，避免过于激进
        
        # 静态区域densification限制 (新增)
        self.limit_static_densify = False  # 是否限制静态区域的clone/split
        self.static_densify_percentile = 25  # 动态指标低于此百分位的高斯被认为是静态的，限制其densification
        
        self.div_percentile = -1  # 散度阈值百分位数，高于此值触发 Clone
        self.curl_percentile = -1  # 旋度阈值百分位数，高于此值触发 Split
        self.div_thresh = 0  # 散度硬阈值，>= 0 时使用硬阈值，< 0 时使用百分位数
        self.curl_thresh = 0  # 旋度硬阈值，>= 0 时使用硬阈值，< 0 时使用百分位数
        self.physics_clone_eta = 0.2  # Clone 时沿速度反方向偏移的系数
        self.physics_split_scale_factor = 2.0  # Split 时缩放因子
        super().__init__(parser, "Optimization Parameters", sentinel)


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
