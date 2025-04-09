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

template<const int WarpSize=WARP_SIZE>
__device__ float warp_reduce_sum_f32(float value) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

// FP32
template<const int BlockSize=BLOCK_SIZE>
__global__ void dot_prod_f32(float* a, float* b, float* out, const int num) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BlockSize + tid;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
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

// FP32 + float4
template <const int BlockSize=BLOCK_SIZE>
__global__ void dot_prod_f32x4(float* a, float* b, float* out, const int num) {
    int tid = threadIdx.x;
    int gid = (blockIdx.x * BlockSize + tid) * 4;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    float4 tmp_a, tmp_b;

    if (gid < num) {
        tmp_a = FLOAT4(a[gid]);
        tmp_b = FLOAT4(b[gid]);
    }
    float prod = tmp_a.x * tmp_b.x + tmp_a.y * tmp_b.y + tmp_a.z * tmp_b.z + tmp_a.w * tmp_b.w;
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
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

// FP16
template<const int WarpSize=WARP_SIZE>
__device__ half warp_reduce_sum_f16(half value) {
    #pragma unroll
    for (int mask = WarpSize; mask > 0; mask >>= 1) {
        value = __hadd(value, __shfl_xor_sync(0xffffffff, value, mask));
    }
    return value;
}

template<const int WarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half value) {
  float val_f32 = __half2float(value);
  #pragma unroll
  for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

template<const int BlockSize=BLOCK_SIZE>
__global__ void dot_prod_f16_f32(half* a, half* b, float* out, const int num) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BlockSize + tid;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    half sum = (gid < num) ? __hmul(a[gid], b[gid]) : __float2half(0.0f);
    
    float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum_f32;
        // printf("half:%f\n", __half2float(sum));
    }
    __syncthreads();
    sum_f32 = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) {
        sum_f32 = warp_reduce_sum_f32<NUM_WARPS>(sum_f32);
    }
    if (tid == 0) {
        atomicAdd(out, sum_f32);
    }
}

// Kernel Launchers
// FP32 kernel
void launch_dot_prod_f32(float* a, float* b, float* out, const int num, dim3 grid, dim3 block) {
    dot_prod_f32<<<grid, block>>>(a, b, out, num);
}

// FP32x4 kernel
void launch_dot_prod_f32x4(float* a, float* b, float* out, const int num, dim3 grid, dim3 block) {
    dot_prod_f32x4<<<grid, block>>>(a, b, out, num);
}

void launch_dot_prod_f16(half* a, half* b, float* out, const int num, dim3 grid, dim3 block) {
    dot_prod_f16_f32<<<grid, block>>>(a, b, out, num);
}