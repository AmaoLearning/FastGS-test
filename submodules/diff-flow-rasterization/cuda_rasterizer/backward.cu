/*
 * diff-flow-rasterization  —  backward.cu
 * Mirrors diff-gaussian-rasterization_fastgs/backward.cu exactly, with:
 *   - computeColorFromSH backward  → computeFlowFromVelocity backward (J^T)
 *   - NUM_CHAFFELS (3)             → FLOW_CHANNELS (2)
 *   - No SH gradient; gradient flows to velocity3D via Jacobian transpose.
 *   - PerGaussianRenderCUDA & renderCUDA backward are structurally identical.
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Backward pass for the Jacobian projection:  velocity3D → flow2D.
//
// Forward:  flow = J · (W · v_world)   where J is 2×3 perspective Jacobian
// Backward: dL/dv_world += W^T · J^T · dL/dflow
//           dL/dmean   += (partial J / partial p_cam) contribution
// ---------------------------------------------------------------------------
__device__ void computeFlowFromVelocityBackward(
	int idx,
	const float3* means,
	const float* velocity3D,
	const float* viewmatrix,
	float focal_x, float focal_y,
	float tan_fovx, float tan_fovy,
	const float* dL_dflow,      // [P, 2]
	float3* dL_dmeans,          // accumulated
	float* dL_dvelocity3D)      // [P, 3] output
{
	float3 p_orig = means[idx];

	// Recompute camera-space position
	float3 t = transformPoint4x3(p_orig, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	float3 v_world = { velocity3D[3 * idx + 0],
	                    velocity3D[3 * idx + 1],
	                    velocity3D[3 * idx + 2] };
	float3 v_cam = transformVec4x3(v_world, viewmatrix);

	float inv_z = 1.0f / t.z;
	float inv_z2 = inv_z * inv_z;
	float inv_z3 = inv_z2 * inv_z;

	float dL_dfu = dL_dflow[idx * 2 + 0];
	float dL_dfv = dL_dflow[idx * 2 + 1];

	// ── Gradient w.r.t. v_cam ──
	// flow_u = fx * (v_cam.x / z - x * v_cam.z / z²)
	// flow_v = fy * (v_cam.y / z - y * v_cam.z / z²)
	float dL_dvcx = dL_dfu * focal_x * inv_z;
	float dL_dvcy = dL_dfv * focal_y * inv_z;
	float dL_dvcz = -dL_dfu * focal_x * t.x * inv_z2
	                -dL_dfv * focal_y * t.y * inv_z2;

	// Propagate to world-space velocity:  dL/dv_world = W^T · dL/dv_cam
	float3 dL_dv_cam = { dL_dvcx, dL_dvcy, dL_dvcz };
	float3 dL_dv_world = transformVec4x3Transpose(dL_dv_cam, viewmatrix);
	dL_dvelocity3D[3 * idx + 0] = dL_dv_world.x;
	dL_dvelocity3D[3 * idx + 1] = dL_dv_world.y;
	dL_dvelocity3D[3 * idx + 2] = dL_dv_world.z;

	// ── Gradient w.r.t. camera-space position (affects mean3D) ──
	// Only computed when dL_dmeans != nullptr (geometry gradients requested).
	// Skipped by default to save compute when only optimizing velocity.
	if (dL_dmeans != nullptr)
	{
		float dL_dtx = x_grad_mul * dL_dfu * (-focal_x * v_cam.z * inv_z2);
		float dL_dty = y_grad_mul * dL_dfv * (-focal_y * v_cam.z * inv_z2);
		float dL_dtz = dL_dfu * (-focal_x * v_cam.x * inv_z2 + 2.0f * focal_x * t.x * v_cam.z * inv_z3)
		             + dL_dfv * (-focal_y * v_cam.y * inv_z2 + 2.0f * focal_y * t.y * v_cam.z * inv_z3);
		float3 dL_dm = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);
		dL_dmeans[idx].x += dL_dm.x;
		dL_dmeans[idx].y += dL_dm.y;
		dL_dmeans[idx].z += dL_dm.z;
	}
}

// ---------------------------------------------------------------------------
// Backward of 2D covariance — identical to original
// ---------------------------------------------------------------------------
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float* cov3D = cov3Ds + 6 * idx;

	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	dL_dmeans[idx] = dL_dmean;
}

// ---------------------------------------------------------------------------
// Backward of 3D covariance — identical to original
// ---------------------------------------------------------------------------
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod,
	const glm::vec4 rot, const float* dL_dcov3Ds,
	glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
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

	glm::mat3 S = glm::mat3(1.0f);
	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;
	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// ---------------------------------------------------------------------------
// preprocessCUDA backward — replaces SH backward with velocity backward
// ---------------------------------------------------------------------------
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const float* velocity3D,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const float* viewmatrix,
	float focal_x, float focal_y,
	float tan_fovx, float tan_fovy,
	const float4* dL_dmean2D,
	const float* dL_dflow,          // [P, 2]
	glm::vec3* dL_dmeans,
	float* dL_dvelocity3D,          // [P, 3]
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// ── Only compute velocity gradient ──
	// Geometry gradients (mean2D projection, cov3D → scale/rotation) are intentionally
	// skipped: the flow rasterizer only optimizes velocity3D, geometry is detached
	// at the Python level.  dL_dmean3D / dL_dcov3D / dL_dscale / dL_drot stay zero.
	if (velocity3D != nullptr)
		computeFlowFromVelocityBackward(
			idx, means, velocity3D, viewmatrix,
			focal_x, focal_y, tan_fovx, tan_fovy,
			dL_dflow,
			nullptr,               // dL_dmeans = nullptr: skip position gradient
			dL_dvelocity3D);
}

// ---------------------------------------------------------------------------
// PerGaussianRenderCUDA backward — mirrors original, FLOW_CHANNELS=2
// Uses FastGS per-bucket warp approach with sampled_T / sampled_ar
// ---------------------------------------------------------------------------
template<uint32_t C>
__global__ void
PerGaussianRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int B,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	const uint32_t* __restrict__ bucket_to_tile,
	const float* __restrict__ sampled_T, const float* __restrict__ sampled_ar,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ flow,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ max_contrib,
	const float* __restrict__ pixel_colors,
	const float* __restrict__ dL_dpixels,
	float4* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dflow
) {
	auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	bool valid_bucket = global_bucket_idx < (uint32_t) B;
	if (!valid_bucket) return;

	bool valid_splat = false;

	uint32_t tile_id, bbm;
	uint2 range;
	int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile, splat_idx_global;

	tile_id = bucket_to_tile[global_bucket_idx];
	range = ranges[tile_id];
	num_splats_in_tile = range.y - range.x;
	bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global = range.x + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
		return;
	}

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	float2 xy = {0.0f, 0.0f};
	float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
	float c[C] = {0.0f};
	if (valid_splat) {
		gaussian_idx = point_list[splat_idx_global];
		xy = points_xy_image[gaussian_idx];
		con_o = conic_opacity[gaussian_idx];
		for (int ch = 0; ch < C; ++ch)
			c[ch] = flow[gaussian_idx * C + ch];
	}

	// ── Only accumulate dL/dflow — geometry gradients (mean2D, conic, opacity) skipped ──
	float Register_dL_dflow[C] = {0.0f};

	// tile metadata
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

	float T;
	float last_contributor;
	float dL_dpixel[C];

	// iterate over all pixels in the tile
	#pragma unroll
	for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
		// SHUFFLING
		T = my_warp.shfl_up(T, 1);
		last_contributor = my_warp.shfl_up(last_contributor, 1);
		for (int ch = 0; ch < C; ++ch) {
			dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
		}

		int idx = i - my_warp.thread_rank();
		const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
		const uint32_t pix_id = W * pix.y + pix.x;
		const float2 pixf = {(float) pix.x, (float) pix.y};
		bool valid_pixel = pix.x < W && pix.y < H;

		// First thread in warp loads per-pixel data
		if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
			T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
			last_contributor = n_contrib[pix_id];
			for (int ch = 0; ch < C; ++ch) {
				dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
			}
		}

		if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
			if (W <= pix.x || H <= pix.y) continue;
			if (splat_idx_in_tile >= last_contributor) continue;

			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) continue;
			const float dchannel_dcolor = alpha * T;

			// dL/dflow_i = Σ_p (alpha_i · T_i) · dL/dpixel
			for (int ch = 0; ch < C; ++ch) {
				Register_dL_dflow[ch] += dchannel_dcolor * dL_dpixel[ch];
			}

			T *= (1.0f - alpha);
		}
	}

	// Atomic accumulation — only flow gradients
	if (valid_splat) {
		for (int ch = 0; ch < C; ++ch) {
			atomicAdd(&dL_dflow[gaussian_idx * C + ch], Register_dL_dflow[ch]);
		}
	}
}

// ---------------------------------------------------------------------------
// BACKWARD::preprocess
// ---------------------------------------------------------------------------
void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const float* velocity3D,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float4* dL_dmean2D,
	const float* dL_dconics,
	const float* dL_dflow,
	glm::vec3* dL_dmean3D,
	float* dL_dvelocity3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// NOTE: computeCov2DCUDA and computeCov3D are intentionally NOT launched.
	// The flow rasterizer only optimizes velocity3D; geometry is detached
	// at the Python level, so dL_dmean3D / dL_dcov3D / dL_dscale / dL_drot stay zero.

	// Preprocess backward (velocity only)
	preprocessCUDA <<<(P + 255) / 256, 256>>> (
		P,
		(float3*)means3D,
		radii,
		velocity3D,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		viewmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(float4*)dL_dmean2D,
		dL_dflow,
		(glm::vec3*)dL_dmean3D,
		dL_dvelocity3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

// ---------------------------------------------------------------------------
// BACKWARD::render — launches PerGaussianRenderCUDA<FLOW_CHANNELS>
// ---------------------------------------------------------------------------
void BACKWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int R, int B,
	const uint32_t* per_bucket_tile_offset,
	const uint32_t* bucket_to_tile,
	const float* sampled_T, const float* sampled_ar,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* flow,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const uint32_t* max_contrib,
	const float* pixel_colors,
	const float* dL_dpixels,
	float4* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dflow)
{
	const int THREADS = 32;
	PerGaussianRenderCUDA<FLOW_CHANNELS> <<<((B*32) + THREADS - 1) / THREADS, THREADS>>>(
		ranges,
		point_list,
		W, H, B,
		per_bucket_tile_offset,
		bucket_to_tile,
		sampled_T, sampled_ar,
		bg_color,
		means2D,
		conic_opacity,
		flow,
		final_Ts,
		n_contrib,
		max_contrib,
		pixel_colors,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dflow);
}
