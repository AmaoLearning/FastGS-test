/*
 * diff-flow-rasterization
 * Adapted from diff-gaussian-rasterization_fastgs.
 * Auxiliary device helpers — identical to the original except:
 *   - SH coefficients removed (not needed for flow)
 *   - BLOCK_SIZE / NUM_WARPS now reference FLOW_CHANNELS-aware config
 */

#ifndef CUDA_FLOW_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_FLOW_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <stdint.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

// ---- Coordinate conversion ------------------------------------------------
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

// ---- Tile rect computation ------------------------------------------------
__forceinline__ __device__ void getRect(const float2 p, int max_radius,
	uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect,
	uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

// ---- Affine transforms (4x3, 4x4) ----------------------------------------
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2]  * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6]  * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

// ---- Norm derivative helpers (unchanged from original) --------------------
__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

// ---- Frustum culling (unchanged) ------------------------------------------
__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// ---- Tile-intersection helpers (identical to original) --------------------
__device__ inline float2 computeEllipseIntersection(
    const float4 con_o, const float disc, const float t, const float2 p,
    const bool isY, const float coord)
{
	float p_u = isY ? p.y : p.x;
	float p_v = isY ? p.x : p.y;
	float coeff = isY ? con_o.x : con_o.z;

	float h = coord - p_u;
	float sqrt_term = sqrt(disc * h * h + t * coeff);

	return {
		(-con_o.y * h - sqrt_term) / coeff + p_v,
		(-con_o.y * h + sqrt_term) / coeff + p_v
	};
}

__device__ inline uint32_t processTiles(
    const float4 con_o, const float disc, const float t, const float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    const dim3 grid, const bool isY,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted)
{
	float BLOCK_U = isY ? BLOCK_Y : BLOCK_X;
	float BLOCK_V = isY ? BLOCK_X : BLOCK_Y;

	if (isY) {
		rect_min = {rect_min.y, rect_min.x};
		rect_max = {rect_max.y, rect_max.x};
		bbox_min = {bbox_min.y, bbox_min.x};
		bbox_max = {bbox_max.y, bbox_max.x};
		bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
		bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
	}

	uint32_t tiles_count = 0;
	float2 intersect_min_line, intersect_max_line;
	float ellipse_min, ellipse_max;
	float min_line, max_line;

	intersect_max_line = {bbox_max.y, bbox_min.y};

	min_line = rect_min.x * BLOCK_U;
	if (bbox_min.x <= min_line) {
		intersect_min_line = computeEllipseIntersection(
			con_o, disc, t, p, isY, rect_min.x * BLOCK_U);
	} else {
		intersect_min_line = intersect_max_line;
	}

	for (int u = rect_min.x; u < rect_max.x; ++u)
	{
		max_line = (u + 1) * BLOCK_U;
		bool max_line_inside = max_line < bbox_max.x;
		if (max_line_inside) {
			intersect_max_line = computeEllipseIntersection(
				con_o, disc, t, p, isY, max_line);
		} else {
			intersect_max_line = {bbox_argmax.y, bbox_argmin.y};
		}

		ellipse_min = fminf(fminf(intersect_min_line.x, intersect_max_line.x),
			bbox_argmin.y);
		ellipse_max = fmaxf(fmaxf(intersect_min_line.y, intersect_max_line.y),
			bbox_argmax.y);

		int v_min = max((int)0, min((int)(isY ? grid.x : grid.y),
			(int)(ellipse_min / BLOCK_V)));
		int v_max = max((int)0, min((int)(isY ? grid.x : grid.y),
			(int)(ellipse_max / BLOCK_V) + 1));

		for (int v = v_min; v < v_max; ++v)
		{
			uint32_t tile_x = isY ? v : u;
			uint32_t tile_y = isY ? u : v;
			uint64_t key = tile_y * grid.x + tile_x;
			key <<= 32;
			key |= *((uint32_t*)&depth);
			gaussian_keys_unsorted[off + tiles_count] = key;
			gaussian_values_unsorted[off + tiles_count] = idx;
			tiles_count++;
		}

		intersect_min_line = intersect_max_line;
	}

	return tiles_count;
}

__device__ inline void duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid, const float mult,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted)
{
	float det = (con_o.x * con_o.z - con_o.y * con_o.y);
	if (det == 0.0f) return;

	float inv_det = 1.0f / det;
	float a11 = con_o.z * inv_det;
	float a22 = con_o.x * inv_det;
	float a12 = -con_o.y * inv_det;

	float threshold = mult * mult;
	float disc = con_o.y * con_o.y - con_o.x * con_o.z;
	float t = threshold * det;

	float sx = sqrtf(a11 * threshold);
	float sy = sqrtf(a22 * threshold);

	float2 bbox_min = {p.x - sx, p.y - sy};
	float2 bbox_max = {p.x + sx, p.y + sy};

	int2 rect_min = {
		max((int)0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
		max((int)0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y)))
	};
	int2 rect_max = {
		max((int)0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X) + 1)),
		max((int)0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y) + 1))
	};

	float2 bbox_argmin, bbox_argmax;
	float min_power = 0.0f;
	float max_power = 0.0f;

	if (disc >= 0) {
		float sq = sqrtf(disc);
		float denom_inv = 1.0f / con_o.x;
		bbox_argmin = {p.x, p.y + (-con_o.y - sq) * denom_inv * 1.0f};
		bbox_argmax = {p.x, p.y + (-con_o.y + sq) * denom_inv * 1.0f};
	} else {
		bbox_argmin = {p.x, p.y};
		bbox_argmax = {p.x, p.y};
	}

	bool use_y = (rect_max.x - rect_min.x) <= (rect_max.y - rect_min.y);

	processTiles(con_o, disc, t, p,
		bbox_min, bbox_max, bbox_argmin, bbox_argmax,
		rect_min, rect_max, grid, use_y,
		idx, off, depth,
		gaussian_keys_unsorted, gaussian_values_unsorted);
}

#define CHECK_CUDA(A, debug) \
A; \
if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
