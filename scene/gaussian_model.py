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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.flow_loss_accum = torch.empty(0)
        self.flow_denom = torch.empty(0)
        self._deform_accum = torch.empty(0)   # (N, 3) accumulated d_xyz
        self._deform_sq_accum = torch.empty(0)   # (N, 3) accumulated d_xyz^2
        self._deform_denom = torch.empty(0)   # (N, 1) count
        # Maximum displacement difference tracking (for dynamic score)
        self._deform_max = torch.empty(0)    # (N, 3) max d_xyz per Gaussian
        self._deform_min = torch.empty(0)    # (N, 3) min d_xyz per Gaussian
        self._deform_tracking_started = False  # flag to start tracking from iter 10000
        self._cluster_labels = None  # (N,) int32 tensor: -1 for static, >=0 for cluster id

        self.optimizer = None
        self.shoptimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deform_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_sq_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")        
        self.tracked_for_fft = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device="cuda")        # Initialize max/min deformation tracking
        self._deform_max = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_min = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_tracking_started = False

    def training_setup(self, training_args, args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.flow_loss_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.flow_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deform_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_sq_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Initialize max/min deformation tracking
        self._deform_max = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_min = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_tracking_started = False

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        sh_l = [{'params': [self._features_rest], 'lr': args.highfeature_lr / 20.0, "name": "f_rest"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add cluster labels if they exist (for clustered deform model)
        if hasattr(self, '_cluster_labels') and self._cluster_labels is not None:
            l.append('cluster_label')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        # Include cluster labels if they exist
        if hasattr(self, '_cluster_labels') and self._cluster_labels is not None:
            cluster_labels = self._cluster_labels.detach().cpu().unsqueeze(1).numpy().astype(np.float32)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, cluster_labels), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, dynamic_only: bool = False):
        if dynamic_only and self._cluster_labels is not None:
            # Only reset opacity for dynamic Gaussians, preserving static ones
            dynamic_mask = self._cluster_labels >= 0
            opacities_capped = inverse_sigmoid(
                torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
            )
            with torch.no_grad():
                self._opacity.data[dynamic_mask] = opacities_capped[dynamic_mask]
            # Selectively reset Adam state for dynamic Gaussians only
            for group in self.optimizer.param_groups:
                if group["name"] == "opacity":
                    state = self.optimizer.state.get(group['params'][0], None)
                    if state is not None:
                        state["exp_avg"][dynamic_mask] = 0.0
                        state["exp_avg_sq"][dynamic_mask] = 0.0
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # Load cluster labels if they exist (for clustered deform model)
        try:
            self._cluster_labels = torch.tensor(np.asarray(plydata.elements[0]['cluster_label']), dtype=torch.int32, device="cuda")
            print("[INFO] Load cluster label successfully!")
        except ValueError:
            print("[WARNNING] Failed to load cluster label! Set to None")
            self._cluster_labels = None

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)
        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 16 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.flow_loss_accum = self.flow_loss_accum[valid_points_mask]
        self.flow_denom = self.flow_denom[valid_points_mask]
        self._deform_accum = self._deform_accum[valid_points_mask]
        self._deform_sq_accum = self._deform_sq_accum[valid_points_mask]
        self._deform_denom = self._deform_denom[valid_points_mask]
        self.tracked_for_fft = self.tracked_for_fft[valid_points_mask]
        self._deform_max = self._deform_max[valid_points_mask] if self._deform_max.numel() > 0 else self._deform_max
        self._deform_min = self._deform_min[valid_points_mask] if self._deform_min.numel() > 0 else self._deform_min
        
        # Prune cluster labels if they exist
        if self._cluster_labels is not None:
            self._cluster_labels = self._cluster_labels[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_cluster_labels=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Extend global deform accumulators for newly added Gaussians
        _n_new = new_xyz.shape[0]
        self._deform_accum = torch.cat([self._deform_accum,
                                         torch.zeros(_n_new, 3, device="cuda")], dim=0)
        self._deform_sq_accum = torch.cat([self._deform_sq_accum,
                            torch.zeros(_n_new, 3, device="cuda")], dim=0)
        self._deform_denom = torch.cat([self._deform_denom,
                                         torch.zeros(_n_new, 1, device="cuda")], dim=0)
        self.tracked_for_fft = torch.cat([self.tracked_for_fft,
                                         torch.zeros(_n_new, dtype=torch.bool, device="cuda")], dim=0)
        self._deform_max = torch.cat([self._deform_max,
                                       torch.zeros(_n_new, 3, device="cuda")], dim=0) if self._deform_max.numel() > 0 else self._deform_max
        self._deform_min = torch.cat([self._deform_min,
                                       torch.zeros(_n_new, 3, device="cuda")], dim=0) if self._deform_min.numel() > 0 else self._deform_min
        
        # Extend or initialize cluster labels for newly added Gaussians
        if new_cluster_labels is not None:
            # Inherit cluster labels from parent Gaussians
            self._cluster_labels = torch.cat([self._cluster_labels, new_cluster_labels], dim=0)
        elif self._cluster_labels is not None:
            # Default to -1 (static) for new Gaussians if no parent labels provided
            self._cluster_labels = torch.cat([self._cluster_labels,
                                               torch.full((_n_new,), -1, dtype=torch.int32, device="cuda")], dim=0)

        self.zero_accums()

    def zero_accums(self):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.flow_loss_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.flow_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # NOTE: do NOT reset _deform_accum/_deform_denom here — they accumulate globally

    def reset_deform_accums(self):
        """Reset deformation history accumulators (call after clustering)."""
        self._deform_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_sq_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deform_max = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_min = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._deform_tracking_started = False

    def start_deform_tracking(self):
        """Start tracking max/min deformation from iteration 10000."""
        self._deform_tracking_started = True

    def add_deform_stats(self, d_xyz: torch.Tensor) -> None:
        """Accumulate per-Gaussian deformation for motion history.
        
        When tracking is started (from iter 10000), also tracks max/min displacement.
        """
        d_xyz_detached = d_xyz.detach()
        self._deform_accum += d_xyz_detached
        self._deform_sq_accum += d_xyz_detached.square()
        self._deform_denom += 1
        
        # Track max/min displacement when tracking is enabled
        if self._deform_tracking_started:
            if self._deform_max.numel() == 0 or self._deform_max.shape[0] != d_xyz_detached.shape[0]:
                self._deform_max = d_xyz_detached.clone()
                self._deform_min = d_xyz_detached.clone()
            else:
                self._deform_max = torch.maximum(self._deform_max, d_xyz_detached)
                self._deform_min = torch.minimum(self._deform_min, d_xyz_detached)

    def get_mean_deform(self) -> torch.Tensor:
        """Return per-Gaussian mean deformation (N, 3). Zero if no history."""
        denom = self._deform_denom.clamp(min=1)
        return self._deform_accum / denom

    def get_deform_var(self) -> torch.Tensor:
        """Return per-Gaussian deformation variance (N, 3). Zero if no history."""
        denom = self._deform_denom.clamp(min=1)
        mean = self._deform_accum / denom
        second_moment = self._deform_sq_accum / denom
        return (second_moment - mean.square()).clamp_min(0.0)

    def get_max_deform_diff(self) -> torch.Tensor:
        """Return per-Gaussian maximum displacement difference (N,) - magnitude of max - min displacement."""
        if self._deform_max.numel() == 0 or self._deform_min.numel() == 0:
            return torch.zeros(self.get_xyz.shape[0], device="cuda")
        # Compute L2 norm of (max - min) for each Gaussian
        diff = self._deform_max - self._deform_min  # (N, 3)
        return torch.norm(diff, dim=-1)  # (N,)

    def compute_dynamic_score(self) -> torch.Tensor:
        """Compute dynamic score as harmonic mean of percentile ranks.
        
        Dynamic score is the harmonic mean of:
        1. Max displacement difference percentile: rank of each gaussian's max displacement diff in percent
        2. Deformation variance percentile: rank of each gaussian's deformation variance in percent
        
        Returns:
            dynamic_score: (N,) tensor with scores in range [0, 1]
        """
        # Check if we have any deformation data at all
        if self._deform_denom.sum().item() == 0:
            print(f"[DEBUG] compute_dynamic_score: no deform data, returning zeros")
            return torch.zeros(self.get_xyz.shape[0], device="cuda")
        
        if not self._deform_tracking_started:
            # Even if tracking not started, we can still use variance-based scoring
            # But warn that max/min tracking is not active
            print(f"[DEBUG] compute_dynamic_score: tracking not started, using variance only")
        
        # Check if max/min tracking has actual data (not all zeros)
        max_diff = self.get_max_deform_diff()  # (N,)
        
        deform_var = self.get_deform_var()  # (N, 3)
        deform_var_sum = deform_var.sum(dim=-1)  # (N,)
        
        # 3. Compute percentile ranks (0-100)
        # searchsorted returns indices (0 to N-1), convert to percentile (0 to 100)
        N_gaussians = max(1, self.get_xyz.shape[0])
        
        # For max displacement difference percentile
        max_diff_sorted, _ = torch.sort(max_diff)
        max_diff_indices = torch.searchsorted(max_diff_sorted, max_diff)  # (N,), returns 0 to N
        max_diff_percentile = (max_diff_indices.float() / max(N_gaussians - 1, 1))  # (N,), 0-100
        
        # For deformation variance percentile
        var_sorted, _ = torch.sort(deform_var_sum)
        var_indices = torch.searchsorted(var_sorted, deform_var_sum)  # (N,), returns 0 to N
        var_percentile = (var_indices.float() / max(N_gaussians - 1, 1))  # (N,), 0-100
        
        # 5. Harmonic mean: 2 * a * b / (a + b)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        harmonic_mean = 2.0 * max_diff_percentile * var_percentile / (max_diff_percentile + var_percentile + epsilon)
        
        return harmonic_mean  # (N,), range [0, 1]

    def get_dynamic_gaussian_mask(self, threshold: float = 0.8) -> torch.Tensor:
        """Get boolean mask for dynamic Gaussians based on dynamic score threshold.
        
        Args:
            threshold: Dynamic score threshold (default 0.8 = 80%)
            
        Returns:
            mask: (N,) boolean tensor, True for dynamic Gaussians
        """
        dynamic_score = self.compute_dynamic_score()
        return dynamic_score > threshold

    def get_dynamic_mask_from_cluster(self) -> torch.Tensor:
        """Get dynamic mask from cluster labels.
        
        Returns:
            dynamic_mask: (N,) bool tensor, True for dynamic Gaussians (cluster >= 0)
        """
        if self._cluster_labels is None:
            return torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        
        return self._cluster_labels >= 0
    
    def get_cluster_mask(self, cluster_id: int) -> torch.Tensor:
        """Get mask for a specific cluster.
        
        Args:
            cluster_id: Cluster ID (0 to n_clusters-1)
        
        Returns:
            mask: (N,) bool tensor, True for Gaussians in the specified cluster
        """
        if self._cluster_labels is None:
            return torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        
        return self._cluster_labels == cluster_id

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                       new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        prune_mask = torch.logical_and(prune_mask, ~self.tracked_for_fft.squeeze())
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # Inherit cluster labels from parent Gaussians
        new_cluster_labels = None
        if self._cluster_labels is not None:
            new_cluster_labels = self._cluster_labels[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_cluster_labels)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        """Fast Gaussian Splatting clone: duplicate small gaussians that have large gradients."""
        selected_pts_mask = torch.logical_and(metric_mask, filter)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        # Inherit cluster labels from parent Gaussians
        new_cluster_labels = None
        if self._cluster_labels is not None:
            new_cluster_labels = self._cluster_labels[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_cluster_labels)

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None, flow_mask = None, dynamic_only: bool = False):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned.
            4.  (New) If flow_mask is provided, only gaussians with accurate flow prediction are densified.
            5.  (New) If dynamic_only is True and cluster labels exist, static Gaussians
                (cluster_label == -1) are excluded from densification and opacity cap.
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_norm = torch.norm(grad_vars, dim=-1)
        grads_abs_norm = torch.norm(grads_abs, dim=-1)

        grad_qualifiers = grad_norm >= args.grad_thresh
        grad_qualifiers_abs = grads_abs_norm >= args.grad_abs_thresh

        # Ablation: lower densification thresholds for dynamic Gaussians
        if self._cluster_labels is not None:
            _dyn = self._cluster_labels >= 0
            if args.dynamic_grad_thresh >= 0:
                grad_qualifiers[_dyn] = grad_norm[_dyn] >= args.dynamic_grad_thresh
            if args.dynamic_grad_abs_thresh >= 0:
                grad_qualifiers_abs[_dyn] = grads_abs_norm[_dyn] >= args.dynamic_grad_abs_thresh

        # Exclude static Gaussians from densification when dynamic_only is set
        if dynamic_only and self._cluster_labels is not None:
            static_mask = self._cluster_labels < 0
            grad_qualifiers[static_mask] = False
            grad_qualifiers_abs[static_mask] = False

        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # Collect all mask counts on GPU first, then do a single GPU→CPU sync
        _stats = [all_clones.sum(), all_splits.sum()]  # accumulate GPU tensors
        
        # 如果提供了 flow_mask，则与原有掩码做与运算
        # flow_mask 为 True 表示该高斯的光流拟合质量好，可以参与 densification
        if flow_mask is not None:
            all_clones = torch.logical_and(all_clones, flow_mask)
            all_splits = torch.logical_and(all_splits, flow_mask)
            _stats.extend([flow_mask.sum(), all_clones.sum(), all_splits.sum()])

        # This is our multi-view consisent metric for densification
        # We use this metric to further filter the candidates for densification, which is similar to taming 3dgs.
        _imp_thresh = 5.0
        if self._cluster_labels is not None and args.dynamic_importance_thresh >= 0:
            # Per-group importance threshold: lower for dynamic, default for static
            metric_mask = torch.zeros_like(grad_qualifiers)
            _dyn_imp = self._cluster_labels >= 0
            metric_mask[_dyn_imp] = importance_score[_dyn_imp] > args.dynamic_importance_thresh
            metric_mask[~_dyn_imp] = importance_score[~_dyn_imp] > _imp_thresh
        else:
            metric_mask = importance_score > _imp_thresh

        _all_clones = torch.logical_and(metric_mask, all_clones)
        _all_splits = torch.logical_and(metric_mask, all_splits)
        _stats.extend([metric_mask.sum(), _all_clones.sum(), _all_splits.sum()])

        # Single GPU→CPU synchronization point for ALL mask statistics
        _vals = torch.stack(_stats).cpu().tolist()
        _i = 0
        print(f"\nOriginal all_clones: {int(_vals[_i])}, all_splits: {int(_vals[_i+1])}")
        _i += 2
        if flow_mask is not None:
            print(f"With flow_mask: {int(_vals[_i])}, all_clones: {int(_vals[_i+1])}, all_splits: {int(_vals[_i+2])}")
            _i += 3
        print(f"With metric_mask: {int(_vals[_i])}, all_clones: {int(_vals[_i+1])}, all_splits: {int(_vals[_i+2])}\n")

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # When dynamic_only, restrict pruning to dynamic Gaussians only.
        # Without this guard, static Gaussians (whose opacity has drifted low
        # due to the lack of opacity resets in the student window) would consume
        # the entire remove_budget, causing net Gaussian count to decrease even
        # though ~3k dynamic Gaussians were just added above.
        if dynamic_only and self._cluster_labels is not None:
            static_mask = self._cluster_labels < 0
            prune_mask[static_mask] = False

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)  # single GPU→CPU sync

        # The budget is not necessary for our method.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            # ALL computation on GPU — avoid GPU→CPU data transfer
            padded_importance = torch.zeros(n_init_points, dtype=torch.float32, device="cuda")
            padded_importance[:scores.shape[0]] = 1.0 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros(n_init_points, dtype=torch.bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            final_prune = torch.logical_and(final_prune, ~self.tracked_for_fft.squeeze())
            self.prune_points(final_prune)
        
        # Cap opacity: only for dynamic Gaussians when dynamic_only is set,
        # preserving static Gaussians' opacity and Adam state.
        if dynamic_only and self._cluster_labels is not None:
            dynamic_mask = self._cluster_labels >= 0
            opacities_capped = inverse_sigmoid(
                torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8)
            )
            with torch.no_grad():
                self._opacity.data[dynamic_mask] = opacities_capped[dynamic_mask]
            # Selectively reset Adam state for dynamic Gaussians only
            for group in self.optimizer.param_groups:
                if group["name"] == "opacity":
                    state = self.optimizer.state.get(group['params'][0], None)
                    if state is not None:
                        state["exp_avg"][dynamic_mask] = 0.0
                        state["exp_avg_sq"][dynamic_mask] = 0.0
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        # NOTE: torch.cuda.empty_cache() removed — it forces a full
        # driver-level cache purge (~1-5ms stall). The caching allocator
        # already reuses freed blocks efficiently.

    def final_prune_fastgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        
        final_prune = torch.logical_and(final_prune, ~self.tracked_for_fft.squeeze())
        self.prune_points(final_prune)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_flow_loss_stats(self, per_gaussian_flow_error, update_filter):
        """累积每个高斯的光流拟合误差（用于 densification 掩码）
        Args:
            per_gaussian_flow_error: shape [N, 1], 每个高斯点的光流误差信号
            update_filter: 可见性掩码
        """
        self.flow_loss_accum[update_filter] += per_gaussian_flow_error[update_filter]
        self.flow_denom[update_filter] += 1

    def get_flow_loss_mask(self, threshold, adaptive_percentile=-1):
        """基于累积的平均光流误差生成掩码
        Args:
            threshold: 光流误差阈值，低于此值的高斯将被标记
            adaptive_percentile: 自适应阈值百分比，-1表示使用固定阈值，0-100表示使用自适应阈值
                                 例如: 70 表示取70分位数作为阈值，使得约70%的高斯通过筛选
        Returns:
            mask: bool tensor, True 表示该高斯的平均光流误差低于阈值（光流拟合较好）
        """
        avg_flow_loss = self.flow_loss_accum / (self.flow_denom + 1e-7)
        avg_flow_loss[avg_flow_loss.isnan()] = 0.0
        avg_flow_loss = avg_flow_loss.squeeze()
        
        # 获取有效高斯的掩码
        valid_mask = self.flow_denom.squeeze() > 0
        
        # 如果使用自适应阈值
        if adaptive_percentile >= 0 and valid_mask.sum() > 0:
            valid_losses = avg_flow_loss[valid_mask]
            # 计算自适应阈值：取指定百分位数
            adaptive_threshold = torch.quantile(valid_losses, adaptive_percentile / 100.0).item()
            threshold = adaptive_threshold
            print(f"  [Adaptive Threshold] Using {adaptive_percentile}th percentile: {threshold:.6f}")
        
        # 打印分布统计信息，方便设置阈值 — single GPU→CPU sync for all stats
        if valid_mask.sum() > 0:
            valid_losses = avg_flow_loss[valid_mask]
            percentiles = [25, 50, 75, 90, 95, 99]
            pct_tensors = [torch.quantile(valid_losses, p / 100.0) for p in percentiles]
            all_stats = torch.stack([
                valid_mask.sum().float(),
                valid_losses.min(), valid_losses.max(),
                valid_losses.mean(), valid_losses.std(),
                (valid_losses < threshold).sum().float(),
            ] + pct_tensors).cpu().tolist()
            _n_valid = int(all_stats[0])
            print(f"\n[Flow Loss Distribution]")
            print(f"  Total Gaussians: {avg_flow_loss.shape[0]}, Valid (with stats): {_n_valid}")
            print(f"  Min: {all_stats[1]:.6f}, Max: {all_stats[2]:.6f}")
            print(f"  Mean: {all_stats[3]:.6f}, Std: {all_stats[4]:.6f}")
            for i, p in enumerate(percentiles):
                print(f"  {p}th percentile: {all_stats[6 + i]:.6f}")
            print(f"  Current threshold: {threshold}")
            print(f"  Gaussians below threshold: {int(all_stats[5])} ({100*all_stats[5]/_n_valid:.2f}%)\n")
        
        # 仅允许有统计样本且光流误差较低的高斯参与 densification
        return torch.logical_and(valid_mask, avg_flow_loss <= threshold)
