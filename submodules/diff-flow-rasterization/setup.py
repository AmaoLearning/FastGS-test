"""
diff-flow-rasterization  —  setup.py
Mirrors diff-gaussian-rasterization_fastgs/setup.py
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_flow_rasterization",
    packages=['diff_flow_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_flow_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                                         "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diff-gaussian-rasterization_fastgs", "third_party/glm/")]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
