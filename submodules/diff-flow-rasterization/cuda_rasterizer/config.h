/*
 * diff-flow-rasterization
 * Adapted from diff-gaussian-rasterization_fastgs for optical flow rendering.
 * Blends 2-channel projected optical flow via transmittance, mirroring
 * the RGB alpha-blending pipeline of the original rasterizer.
 */

#ifndef CUDA_FLOW_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_FLOW_RASTERIZER_CONFIG_H_INCLUDED

#define FLOW_CHANNELS 2  // (flow_u, flow_v) instead of RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#endif
