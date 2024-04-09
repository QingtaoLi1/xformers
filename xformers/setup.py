#!/usr/bin/env python3

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
from sysconfig import get_paths as gp


compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 10 + compute_capability[1]

python_include_path = gp()['include']
cuda_path = os.environ["CUDA_ROOT"]
torch_path = torch.__path__[0]
cutlass_path = '/home/qingtaoli/cutlass/'

xformers_cmd_as_reference = '''
/usr/local/cuda/bin/nvcc -I/home/qingtaoli/xformers/xformers/csrc/swiglu -I/home/qingtaoli/xformers/third_party/sputnik -I/home/qingtaoli/xformers/third_party/cutlass/include -I/home/qingtaoli/xformers/third_party/cutlass/examples -I/anaconda/envs/dsl/lib/python3.9/site-packages/torch/include -I/anaconda/envs/dsl/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/anaconda/envs/dsl/lib/python3.9/site-packages/torch/include/TH -I/anaconda/envs/dsl/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda/include -I/anaconda/envs/dsl/include/python3.9 -c xformers/csrc/swiglu/cuda/dual_gemm_silu_identity_mul.cu -o build/temp.linux-x86_64-cpython-39/xformers/csrc/swiglu/cuda/dual_gemm_silu_identity_mul.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -DHAS_PYTORCH --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --extended-lambda -D_ENABLE_EXTENDED_ALIGNED_STORAGE -std=c++17 --generate-line-info -DNDEBUG --threads 4 --ptxas-options=-v --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80
'''

setup(
    name='swiglu_extension',
    ext_modules=[
        CUDAExtension(
            name='swiglu_extension',
            sources=[
                'swiglu/swiglu_packedw.cpp',
                'swiglu/swiglu_op.cpp',
                'swiglu/cuda/dual_gemm_silu_identity_mul.cu',
                'swiglu/cuda/gemm_fused_operand_sum.cu',
                'swiglu/cuda/silu_bw_fused.cu',
            ],
            include_dirs=[
                os.path.join(cutlass_path, 'include'),
                os.path.join(cutlass_path, 'examples'),
                # '/home/qingtaoli/cutlass/tools/util/include',
                # '/home/qingtaoli/cutlass/examples/common',
                os.path.join(torch_path, 'include'),
                os.path.join(torch_path, 'include', 'torch', 'csrc', 'api', 'include'),
                os.path.join(torch_path, 'include', 'TH'),
                os.path.join(torch_path, 'include', 'THC'),
                os.path.join(cuda_path, 'include'),
                python_include_path,
            ],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3',
                ],
                'nvcc': [
                    '-std=c++17',
                    # '-O3',
                    '-DHAS_PYTORCH',
                    # '-DBUILD_PYTHON_PACKAGE',
                    '-D_ENABLE_EXTENDED_ALIGNED_STORAGE',
                    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                    '-DTORCH_API_INCLUDE_EXTENSION_H',
                    '-DPYBIND11_COMPILER_TYPE=\"_gcc\"',
                    '-DPYBIND11_STDLIB=\"_libstdcpp\"',
                    '-DPYBIND11_BUILD_ABI=\"_cxxabi1011\"',
                    # '-DTORCH_EXTENSION_NAME=_C',
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '--extended-lambda',
                    f'-gencode=arch=compute_{cuda_arch},code=compute_{cuda_arch}',
                    f'-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}',
                    '--compiler-options="-fPIC"',
                    # '--shared',
                    '--threads=4',
                    '--ptxas-options=-v',
                    '--ptxas-options=-O2',
                    '--ptxas-options=-allow-expensive-optimizations=true',
                ]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


