#include <stdio.h>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "block_reduce.cuh"


#define INT4(x) (reinterpret_cast<int4*>(&(x))[0])
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
#define HALF2(x) (reinterpret_cast<half2*>(&(x))[0])
#define HALF8
#define BFLOAT2(x) (reinterpret_cast<__nv_bfloat162*>(&(x))[0])
#define LDST128BITS(x) (reinterpret_cast<float4*>(&(x))[0])

/*********** FP32 ***********/
template <const int WarpSize>
__device__ __forceinline__ float warp_reduce_f32_kernel(float val) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

template <const int BlockSize>
__global__ void block_reduce_f32_kernel(float* input, float* output, const int n) {
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
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
    if (tid == 0) {
        atomicAdd(output, sum);
    }
}

// FP32 + float4
template<const int BlockSize>
__global__ void block_reduce_f32x4_kernel(float* input, float* output, const int n) {
    int tid = threadIdx.x;
    int gid = (blockIdx.x * BlockSize + tid) * 4;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float4 ele = FLOAT4(input[gid]);
    float sum = (gid < n) ? (ele.x + ele.y + ele.z + ele.w) : 0.0f;
    sum = warp_reduce_f32_kernel<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        sum = warp_reduce_f32_kernel<NUM_WARPS>(sum);
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }

}

/********* FP16 *********/
template<const int WarpSize>
__device__ __forceinline__ half warp_reduce_f16_kernel(half val) {
  #pragma unroll
  for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int BlockSize>
__global__ void block_reduce_f16_kernel(half* input, float* output, const int n) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * BlockSize + tid;
  constexpr int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enough for warp operaion.
  half sum_f16 = (gid < n) ? input[gid] : __float2half(0.0f);
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_f16 = warp_reduce_f16_kernel<WARP_SIZE>(sum_f16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce 
  // with fp32 inter warps.
  if (lane_id == 0) {
    reduce_smem[warp_id] = __half2float(sum_f16);
    }
  __syncthreads(); // make sure the data is in shared memory.

  if (warp_id == 0) {
    float sum = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : 0.0f;
    sum = warp_reduce_f32_kernel<NUM_WARPS>(sum);
    if (lane_id == 0) {
        atomicAdd(output, sum);
    }
  }
}

// FP16 + half2
template <const int BlockSize>
__global__ void block_reduce_f16x2_kernel(half* input, float* output, const int n) {
    int tid = threadIdx.x;
    int gid = (blockIdx.x * BlockSize + tid) * 2;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    half2 reg = HALF2(input[gid]);
    half sum_f16 = (gid < n) ? __hadd(reg.x, reg.y) : __float2half(0.0f);
    
    sum_f16 = warp_reduce_f16_kernel<WARP_SIZE>(sum_f16);
    if (lane_id == 0) {
        smem[warp_id] = __half2float(sum_f16);
    }
    __syncthreads();

    // perform fp32 warp reduce on the first warp
    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        sum = warp_reduce_f32_kernel<NUM_WARPS>(sum);
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

// FP16 + half8
template <const int BlockSize>
__global__ void block_reduce_f16x8_kernel(half* input, float* output, const int n) {
    int tid = threadIdx.x;
    int gid = (blockIdx.x * BlockSize + tid) * 8;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    half pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(input[gid]);
    float sum_f32 = 0.0f;

    // using fp16 would cause overflow in fp16x8
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum_f32+= (((gid + i) < n) ? __half2float(pack[i]) : 0.0f);
    }
    sum_f32 = warp_reduce_f32_kernel<WARP_SIZE>(sum_f32);
    if (lane_id == 0) {
        smem[warp_id] = sum_f32;
    }
    __syncthreads();
    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        sum = warp_reduce_f32_kernel<NUM_WARPS>(sum);
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

/********* BF16 *********/
template<const int WarpSize>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_bf16_kernel(__nv_bfloat16 val) {
    #pragma unroll
    for (int mask = warpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<const int BlockSize>
__global__ void block_reduce_bf16_kernel(__nv_bfloat16 *input, float *output, const int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BlockSize + tid;
    const int NUM_WARPS = (BlockSize + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    __shared__ __nv_bfloat16 smem[NUM_WARPS];

    __nv_bfloat16 sum_bf16 = (gid < n) ? input[gid] : __float2bfloat16(0.0f);

    sum_bf16 = warp_reduce_bf16_kernel<WARP_SIZE>(sum_bf16);
    if (lane_id == 0) {
        // smem[warp_id] = __bfloat162float(sum_bf16);
        smem[warp_id] = sum_bf16;
    }
    __syncthreads();
    if (warp_id == 0) {
        __nv_bfloat16 sum = (lane_id < NUM_WARPS) ? smem[lane_id] : __float2bfloat16(0.0f);
        sum = warp_reduce_bf16_kernel<WARP_SIZE>(sum);
        if (tid == 0) {
            atomicAdd(output,  __bfloat162float(sum));
        }
    }
}

/********* kernel launchers *********/
void launch_block_reduce_f32(float* input, float* output, const int num, dim3 grid, dim3 block) {
    block_reduce_f32_kernel<<<grid, block>>>(input, output, num);
}

void launch_block_reduce_f32x4(float* input, float* output, const int num, dim3 grid, dim3 block) {
    block_reduce_f32x4_kernel<<<grid, block>>>(input, output, num);
}

void launch_block_reduce_f16(half* input, float* output, const int num, dim3 grid, dim3 block) {
    block_reduce_f16_kernel<<<grid, block>>>(input, output, num);
}

void launch_block_reduce_f16x2(half* input, float* output, const int num, dim3 grid, dim3 block) {
    block_reduce_f16x2_kernel<<<grid, block>>>(input, output, num);
}

void launch_block_reduce_f16x8(half* input, float* output, const int num, dim3 grid, dim3 block) {
    block_reduce_f16x8_kernel<<<grid, block>>>(input, output, num);
}

void launch_block_reduce_bf16(__nv_bfloat16 *input, float *output, const int num, dim3 grid, dim3 block) {
    block_reduce_bf16_kernel<<<grid, block>>>(input, output, num);
}