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

        # deformation network type: "mlp" (original) or "4dgs" (HexPlane)
        self.deform_type = "mlp"

        # HexPlane architecture (only effective when deform_type="4dgs")
        # Comma-separated resolutions, e.g. "64,128,256" → 3 levels
        self.hex_spatial_res = "64,128,256"
        self.hex_time_res = "64,128,256"
        self.hex_feat_dim = 16
        self.hex_mlp_hidden = 162
        self.hex_mlp_layers = 2
        self.hex_fusion = "concat"

        # flow-guided losses
        self.use_flow_loss = False     # 启用光流引导的渲染损失
        self.use_flow_mask = False     # 启用光流掩码（仅动态区域计算损失）

        # soft dynamic-static separation
        self.use_dynamic_sep = True   # 启用动静分离：积累形变统计、执行聚类（默认启用）
        # When True, skip clustering entirely and train the 4DGS teacher field end-to-end.
        # use_dynamic_sep can remain True (deform stats are still accumulated for analysis),
        # but the switch to ClusteredDeformModel is suppressed.
        self.teacher_only = False
        self.log_deform_hist = False   # 每隔3000轮记录全局形变分布柱状图

        # Dynamic Gaussian clustering (run at 15000 to assign deform fields, labels persist with Gaussians)
        self.clustering_iterations = "15000" # Comma-separated iteration numbers, e.g. "15000,30000" → [15000, 30000]
        self.cluster_n_clusters = 8        # KMeans 聚类数
        self.cluster_dynamic_thresh = 0.5  # 动态概率阈值，仅 prob > thresh 的高斯参与聚类（用于旧版本兼容）
        self.cluster_w_xyz = 1.0           # 聚类特征权重：3D 位置
        self.cluster_w_color = 0.1         # 聚类特征权重：SH 0阶 (RGB)
        self.cluster_w_motion = 0.5        # 聚类特征权重：历史平均形变
        self.cluster_aabb_padding = 0.15   # 逐簇 AABB 安全边距（相对于簇的位移感知范围）

        # mapo
        self.dynamic_score_percentile = 80  # 动态分数百分位阈值：选取动态分数位于前 (100-percentile)% 的高斯参与聚类，默认为 80 表示前 20%
        
        # Clustered multi-deform model (teacher-student distillation)
        self.clustered_deform_start_iter = 15000  # 多形变场训练起始轮次
        self.teacher_checkpoint_path = ""  # 外部预训练教师模型权重路径（可选；为空则使用当前训练中的形变场作为教师）
        self.use_batched_students = False  # 使用 BatchedHexPlaneDeformNetwork (Path A+B 并行化) — 适用于均匀高tier配置的scaling实验
        
        # Student model architecture parameters (for clustered deform model)
        self.student_feat_dim = 8  # Student HexPlane feature dimension (default: 8)
        self.student_spatial_res = "64,128"  # Student spatial resolutions, comma-separated (default: 2 levels)
        self.student_time_res = "64,128"  # Student temporal resolutions, comma-separated
        self.student_mlp_hidden = 64  # Student MLP hidden dimension
        self.student_mlp_layers = 2  # Student MLP layers
        
        # Warm initialization (teacher → student knowledge transfer)
        self.warm_init_enabled = False
        self.warm_init_downsample_planes = False
        self.warm_init_interpolation_mode = "bilinear"
        self.warm_init_feat_method = "pca"  # none / truncate / pca / random_proj
        self.warm_init_transfer_mlp = False
        self.warm_init_normalize_scale = False
        self.warm_init_noise_std = 1e-4
        
        # Capacity allocation parameters (for dynamic score-based allocation)
        self.capacity_allocation_strategy = "tiered"  # "tiered", "linear", or "frequency"
        self.capacity_tier_boundaries = "0.33,0.67"  # Comma-separated boundaries for tiered strategy
        self.capacity_tier_config_path = "arguments/capacity_tier_configs.json"  # Path to tier config JSON
        self.min_capacity_spatial = "64,96"  # Minimum spatial resolutions
        self.max_capacity_spatial = "64,128,192"  # Maximum spatial resolutions
        self.min_capacity_time = "64,96"  # Minimum temporal resolutions
        self.max_capacity_time = "64,128,192"  # Maximum temporal resolutions
        self.min_capacity_mlp_hidden = 48  # Minimum MLP hidden dimension
        self.max_capacity_mlp_hidden = 96  # Maximum MLP hidden dimension
        self.min_capacity_feat_dim = 8  # Minimum feature dimension
        self.max_capacity_feat_dim = 12  # Maximum feature dimension
        self.capacity_budget_constraint = "none"  # "none", "sum", "max_per_student"
        
        # Frequency-based capacity analysis (only effective when capacity_allocation_strategy="frequency")
        self.freq_n_time_samples = 32  # Number of uniform time steps for teacher sampling

        # lazy loading — N3V large-scale dataset (zero OOM)
        self.lazy_load = False
        self.num_images = 300  # number of frames per camera for N3V dataset
        self.lazy_num_workers = 8
        self.lazy_prefetch_factor = 6
        self.lazy_image_buffer_count = 4  # number of persistent GPU image buffers (>=2)
        self.lazy_prefetch_flow_to_cache = False  # use lazy DataLoader order to prewarm flow cache

        self.enable_flow_preload_cache = False  # cache flow tensors on CPU to reduce repeated npy IO
        self.flow_preload_cache_size = 64       # max number of cameras cached for flow preload (0 disables)
        self.flow_preload_cache_device = "cuda"  # preload cache device: "cpu" or "cuda"/"cuda:0"
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

        # HexPlane learning rates (only effective when deform_type="4dgs")
        self.hex_plane_lr_init = 0.02
        self.hex_plane_lr_final = 0.002
        self.hex_mlp_lr_init = 0.001
        self.hex_mlp_lr_final = 0.00001
        
        # Ablation: differentiated densification thresholds for dynamic Gaussians
        # When >= 0, override the default threshold for dynamic Gaussians (cluster_label >= 0).
        # When < 0 (default), fall back to the global grad_thresh / grad_abs_thresh / importance=5.
        self.dynamic_grad_thresh = -1.0        # clone gradient threshold for dynamic Gaussians
        self.dynamic_grad_abs_thresh = -1.0    # split gradient threshold for dynamic Gaussians
        self.dynamic_importance_thresh = -1.0  # multi-view importance threshold for dynamic Gaussians

        # Student phase densification: an independent densification window that runs
        # exclusively for dynamic Gaussians (cluster_label >= 0) after clustering.
        # This window is separate from — and runs after — the global densification
        # window.  Static Gaussians (cluster_label == -1) are never touched here.
        #
        # To enable: set student_densify_until_iter >= 0.
        # student_densify_from_iter = -1 means "start as soon as cluster labels are
        # available", i.e. immediately after the clustering iteration.
        # When student_densify_until_iter < 0 (default), the window is disabled.
        self.student_densify_from_iter = -1     # start of student densification window
        self.student_densify_until_iter = -1    # end of student densification window (master switch: -1 = off)
        self.student_densification_interval = 500  # densification interval (iters) for student phase

        # Ablation: dynamic opacity regularisation.
        # Penalises low opacity on dynamic Gaussians (cluster_label >= 0) after
        # clustering to encourage them to stay opaque and preserve fine details.
        # Loss = weight * mean((1 - opacity_dynamic)^2).  0.0 = disabled.
        self.dynamic_opacity_reg_weight = 0.0
        self.dynamic_opacity_reg_start_iter = 15000

        # HexPlane temporal TV annealing (only effective when deform_type="4dgs")
        # When enabled, the TV loss on the temporal axis of XT/YT/ZT planes is
        # annealed from weight_init down to weight_final over [anneal_start, anneal_end].
        # This allows the deformation field to model large/sudden displacements in the
        # later training phase while remaining stable during warm-up.
        self.hex_tv_temporal_anneal = False           # master switch
        self.hex_tv_temporal_weight_init = 1e-3      # initial temporal TV weight (same as default)
        self.hex_tv_temporal_weight_final = 1e-5     # final temporal TV weight after annealing
        self.hex_tv_temporal_anneal_start = 15000    # iteration to begin annealing
        self.hex_tv_temporal_anneal_end = 30000      # iteration when annealing completes

        # Boundary regularization (Module C)
        # Penalises deformation magnitude for Gaussians near or beyond the edge of
        # their cluster AABB, preventing grid_sample border-clamping artefacts.
        # Only active when use_dynamic_sep=True and after clustered_deform_start_iter.
        self.boundary_reg_weight = 0.0         # 0.0 = disabled; try 1e-3 ~ 1e-2
        self.boundary_reg_margin = 0.05        # |norm_coord| > (1-margin) → penalised
        self.boundary_reg_start_iter = 15000   # start after clustering
        self.boundary_reg_interval = 10        # compute every N iterations (cost reduction)

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
