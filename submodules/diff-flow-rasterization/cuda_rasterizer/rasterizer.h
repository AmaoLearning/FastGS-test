/*
 * diff-flow-rasterization
 * Rasterizer class interface — mirrors diff-gaussian-rasterization_fastgs/rasterizer.h
 * Adapted for 2-channel flow output with velocity3D input.
 */

#ifndef CUDA_FLOW_RASTERIZER_H_INCLUDED
#define CUDA_FLOW_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaFlowRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int, int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* velocity3D,         // [P, 3]  world-space velocity
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float mult,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_flow,                 // [2, H, W]
			float* out_depth,                // [1, H, W]
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int R, int B,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* velocity3D,         // [P, 3]
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			char* sample_buffer,
			const float* dL_dpix,            // [2, H, W]
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dflow,                 // [P, 2]
			float* dL_dmean3D,
			float* dL_dvelocity3D,           // [P, 3]
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif
