#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dot_product_kernel.cuh"

#define INT4(x) (reinterpret_cast<int4*>(&(x))[0])
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
#define HALF2(x) (reinterpret_cast<half2*>(&(x))[0])
#define BFLOAT2(x) (reinterpret_cast<__nv_bfloat162*>(&(x))[0])
#define LDST128BITS(x) (reinterpret_cast<float4*>(&(x))[0])

template<const int WarpSize=WARP_SZIE>
__device__ float warp_reduce_sum_f32(float value) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

template<const int BlockSize=BLOCK_SIZE>
__global__ void dot_prod_f32(float* a, float* b, float* out, const int num) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BlockSize + tid;
    const int NUM_WARPS = (BlockSize + WARP_SZIE - 1) / WARP_SZIE;
    int warp_id = tid / WARP_SZIE;
    int lane_id = tid % WARP_SZIE;
    __shared__ float smem[NUM_WARPS];

    float prod = (gid < num) ? a[gid] * b[gid] : 0.0f;
    prod = warp_reduce_sum_f32(prod);
    if (lane_id == 0) {
        smem[warp_id] = prod;
        // printf("prod:%f\n", prod);
    }
    __syncthreads();
    prod = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }
    if (tid == 0) {
        atomicAdd(out, prod);
    }
}



// Kernel Launchers
void launch_dot_prod_f32(float* a, float* b, float* out, const int num, dim3 grid, dim3 block) {
    dot_prod_f32<<<grid, block>>>(a, b, out, num);
}