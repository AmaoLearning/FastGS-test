/*
 * diff-flow-rasterization  —  ext.cpp
 * pybind11 module, mirrors diff-gaussian-rasterization_fastgs/ext.cpp
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_flow", &RasterizeFlowCUDA);
  m.def("rasterize_flow_backward", &RasterizeFlowBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
