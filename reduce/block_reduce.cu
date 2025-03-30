#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "block_reduce.cuh"


#define INT4(x) (reinterpret_cast<int4*>(&(x))[0])
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
#define HALF2(x) (reinterpret_cast<half2*>(&(x))[0])
#define BFLOAT2(x) (reinterpret_cast<__nv_bfloat162*>(&(x))[0])
#define LDST128BITS(x) (reinterpret_cast<float4*>(&(x))[0])

template <const int WarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_f32_kernel(float val) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

template <const int BlockSize = BLOCK_SIZE>
__global__ void block_reduce_f32_kernel(float* input, float* output, const int n) {
    int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    int tid = threadIdx.x;
    int gid = blockIdx.x * BlockSize + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    float val;
    __shared__ float shared[NUM_WARPS];
    float sum;

    val = (gid < n) ? input[gid] : 0.0f;
    sum = warp_reduce_f32_kernel<WARP_SIZE>(val);
    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();
    // the first warp compute the final reduce sum.
    val = (warp_id == 0) ? shared[lane_id] : 0.0f;
    sum = warp_reduce_f32_kernel<WARP_SIZE>(val);
    if (gid == 0) {
        atomicAdd(output, sum);
    }
}