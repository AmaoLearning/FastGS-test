/*
 * diff-flow-rasterization
 * Forward pass declarations — mirrors diff-gaussian-rasterization_fastgs/forward.h
 * but replaces SH/color with velocity-to-flow Jacobian projection.
 */

#ifndef CUDA_FLOW_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_FLOW_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to flow rasterization.
	// Compared to the original:
	//   - Replaces SH → RGB with  velocity3D → flow2D  via Jacobian projection.
	//   - Does NOT take dc / shs / degree / cam_pos (no SH evaluation).
	//   - Takes velocity3D (P×3) as input; outputs flow2D (P×2).
	void preprocess(
		int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* velocity3D,        // [P, 3]  world-space velocity
		const float* cov3D_precomp,
		const float* flow_precomp,       // [P, 2]  optional precomputed flow
		const float* viewmatrix,
		const float* projmatrix,
		const float mult,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* flow,                     // [P, 2]  output per-Gaussian flow
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method — identical structure to original render(),
	// but features are 2-channel flow instead of 3-channel RGB.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar,
		int W, int H,
		const float2* points_xy_image,
		const float* features,           // [P, 2]  per-Gaussian 2D flow
		const float* depths,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		float* pixel_colors,             // reused name, stores per-pixel flow
		const float* bg_color,
		float* out_flow,                 // [2, H, W]
		float* out_depth);               // [1, H, W]
}

#endif
