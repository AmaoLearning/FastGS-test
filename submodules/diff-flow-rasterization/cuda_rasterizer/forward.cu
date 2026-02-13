/*
 * diff-flow-rasterization  —  forward.cu
 * Mirrors diff-gaussian-rasterization_fastgs/forward.cu exactly, with:
 *   - computeColorFromSH  →  computeFlowFromVelocity (Jacobian v3D→v2D)
 *   - NUM_CHAFFELS (3)    →  FLOW_CHANNELS (2)
 *   - No sigmoid / SH clamping; flow values are unbounded.
 *   - renderCUDA & bucket/sampled_T/sampled_ar logic are structurally identical.
 */

#include "forward.h"
#include "auxiliary.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Project world-space velocity to image-space 2D optical flow via the
// perspective projection Jacobian.
//
// Given a Gaussian at position p_world with velocity v_world (3D):
//   1) Transform to camera frame:  p_cam = W · p_world,  v_cam = W · v_world
//   2) Project via Jacobian:
//        J = [ fx/z    0    -fx·x/z²  ]
//            [  0    fy/z   -fy·y/z²  ]
//      flow_2d = J · v_cam   (2-vector: du, dv in pixels)
//
// Returns float2 { flow_u, flow_v }.
// ---------------------------------------------------------------------------
__device__ float2 computeFlowFromVelocity(
	int idx,
	const float3* means,
	const float* velocity3D,    // [P, 3]
	const float* viewmatrix,
	float focal_x, float focal_y,
	float tan_fovx, float tan_fovy)
{
	float3 p_orig = means[idx];

	// Transform position to camera space
	float3 t = transformPoint4x3(p_orig, viewmatrix);

	// Clamp to FoV limits (same as computeCov2D)
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	t.x = min(limx, max(-limx, t.x / t.z)) * t.z;
	t.y = min(limy, max(-limy, t.y / t.z)) * t.z;

	// Read world-space velocity
	float3 v_world = { velocity3D[3 * idx + 0],
	                    velocity3D[3 * idx + 1],
	                    velocity3D[3 * idx + 2] };

	// Rotate velocity to camera frame (no translation for vectors)
	float3 v_cam = transformVec4x3(v_world, viewmatrix);

	// Perspective Jacobian
	float inv_z = 1.0f / t.z;
	float inv_z2 = inv_z * inv_z;

	float flow_u = focal_x * (v_cam.x * inv_z - t.x * v_cam.z * inv_z2);
	float flow_v = focal_y * (v_cam.y * inv_z - t.y * v_cam.z * inv_z2);

	return { flow_u, flow_v };
}

// ---------------------------------------------------------------------------
// 2D covariance — identical to original
// ---------------------------------------------------------------------------
__device__ float3 computeCov2D(const float3& mean,
	float focal_x, float focal_y,
	float tan_fovx, float tan_fovy,
	const float* cov3D, const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// ---------------------------------------------------------------------------
// 3D covariance from scale/rotation — identical to original
// ---------------------------------------------------------------------------
__device__ void computeCov3D(const glm::vec3 scale, float mod,
	const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// ---------------------------------------------------------------------------
// preprocessCUDA — mirrors original; replaces SH→RGB with velocity→flow
// ---------------------------------------------------------------------------
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* velocity3D,       // [P, 3] world-space velocity
	const float* viewmatrix,
	const float* projmatrix,
	const float mult,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* flow,                   // [P, C] output per-Gaussian flow
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Near-plane culling
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Project to screen
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// 3D covariance from scales + rotations (always computed)
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	const float* cov3D = cov3Ds + idx * 6;

	// 2D covariance
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Bounding radius
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

	float4 con_o = { conic.x, conic.y, conic.z, opacities[idx] };
	uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, mult, 0, 0, 0, nullptr, nullptr);
	if (tiles_count == 0)
		return;

	// ── velocity → flow via Jacobian projection ─────────────────────
    float2 f = computeFlowFromVelocity(
        idx,
        (const float3*)orig_points,
        velocity3D,
        viewmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy);
    flow[idx * C + 0] = f.x;
    flow[idx * C + 1] = f.y;

	// Store helpers
	depths[idx] = p_view.z;
	radii[idx] = (int)my_radius;
	points_xy_image[idx] = point_image;
	conic_opacity[idx] = con_o;
	tiles_touched[idx] = tiles_count;
}

// ---------------------------------------------------------------------------
// renderCUDA — structurally identical to the original.
// Template on CHANNELS = FLOW_CHANNELS (2).
// Accumulates flow via transmittance: F[ch] += feature[ch] * alpha * T
// No sigmoid — flow values are signed and unbounded.
// ---------------------------------------------------------------------------
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset, uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T, float* __restrict__ sampled_ar,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	float* __restrict__ pixel_colors,
	const float* __restrict__ bg_color,
	float* __restrict__ out_flow,
	float* __restrict__ out_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Bucket-to-tile mapping (for FastGS backward)
	uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	int num_buckets = (toDo + 31) / 32;
	for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
		int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
		if (bucket_idx < num_buckets) {
			bucket_to_tile[bbm + bucket_idx] = tile_id;
		}
	}

	// Shared memory for batch fetching
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Per-pixel state
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D_acc = 0.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Store per-bucket T and accumulated values for FastGS backward
			if (j % 32 == 0) {
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
				for (int ch = 0; ch < CHANNELS; ++ch) {
					sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
				}
				++bbm;
			}

			contributor++;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// ── Alpha-blend flow channels, exactly like RGB ──
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			D_acc += depths[collected_id[j]] * alpha * T;

			T = test_T;
			last_contributor = contributor;
		}
	}

	// Write output
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
		{
			pixel_colors[ch * H * W + pix_id] = C[ch];
			out_flow[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D_acc;
	}

	// Max-reduce last_contributor for FastGS backward
	typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0) {
		max_contrib[tile_id] = last_contributor;
	}
}

// ---------------------------------------------------------------------------
// FORWARD::render  — launch renderCUDA<FLOW_CHANNELS>
// ---------------------------------------------------------------------------
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
	float* sampled_T, float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* features,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	float* pixel_colors,
	const float* bg_color,
	float* out_flow,
	float* out_depth)
{
	renderCUDA<FLOW_CHANNELS> <<<grid, block>>> (
		ranges,
		point_list,
		per_tile_bucket_offset, bucket_to_tile,
		sampled_T, sampled_ar,
		W, H,
		means2D,
		features,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		max_contrib,
		pixel_colors,
		bg_color,
		out_flow,
		out_depth);
}

// ---------------------------------------------------------------------------
// FORWARD::preprocess  — launch preprocessCUDA<FLOW_CHANNELS>
// ---------------------------------------------------------------------------
void FORWARD::preprocess(
	int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* velocity3D,
	const float* viewmatrix,
	const float* projmatrix,
	const float mult,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* flow,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<FLOW_CHANNELS> <<<(P + 255) / 256, 256>>> (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		velocity3D,
		viewmatrix,
		projmatrix,
		mult,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		flow,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}
