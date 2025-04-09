#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

void launch_dot_prod_f32(float* a, float* b, float* out, const int num, dim3 grid, dim3 block);

void launch_dot_prod_f32x4(float* a, float* b, float* out, const int num, dim3 grid, dim3 block);

void launch_dot_prod_f16(half* a, half* b, float* out, const int num, dim3 grid, dim3 block);