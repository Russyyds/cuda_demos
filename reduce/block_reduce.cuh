#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = 256;
}

template <const int BlockSize = BLOCK_SIZE>
__global__ void block_reduce_f32_kernel(float* input, float* output, const int n);

template <const int BlockSize = BLOCK_SIZE>
__global__ void block_reduce_f16_kernel(half* input, float* output, const int n);

template<const int BlockSize = BLOCK_SIZE>
__global__ void block_reduce_bf16_kernel(__nv_bfloat16* input, float* output, const int n);

void launch_block_reduce_f32(float* input, float* output, const int num, dim3 grid, dim3 block);

void launch_block_reduce_f16(half* input, float* output, const int num, dim3 grid, dim3 block);

void launch_block_reduce_bf16(__nv_bfloat16* input, float* output, const int num, dim3 grid, dim3 block);