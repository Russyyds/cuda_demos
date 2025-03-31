// naive safe softmax
// 3 pass
#include <stdio.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "softmax_kernel.cuh"

namespace {
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
}

// warp reduce max
template <typename DATA_TYPE, const int WarpSize = WARP_SIZE>
__device__ __forceinline__ DATA_TYPE warp_reduce_sum(DATA_TYPE val) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask >= 1; mask >>=1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// warp reduce sum
template <typename DATA_TYPE, const int WarpSize = WARP_SIZE>
__device__ __forceinline__ DATA_TYPE warp_reduce_max(DATA_TYPE val) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask >= 1; mask >>=1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template <typename DATA_TYPE, const int NUM_THREADS = 256>
__device__ DATA_TYPE block_reduce_sum(DATA_TYPE val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ DATA_TYPE shared[NUM_WARPS];

    DATA_TYPE value = warp_reduce_sum<DATA_TYPE>(val);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    value = warp_reduce_sum<DATA_TYPE>(value);
    // WRAN: need to broadcast value to all threads within warp
    value = __shfl_sync(0xffffffff, value, 0, WARP_SIZE);
    return value;
}

template<typename DATA_TYPE, const int NUM_THREADS = 256>
__device__ DATA_TYPE block_reduce_max(DATA_TYPE val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __device__ __shared__ DATA_TYPE shared[NUM_WARPS];

    DATA_TYPE value = warp_reduce_max(val);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = (lane < NUM_WARPS) ? shared[lane] : - FLT_MAX;
    value = warp_reduce_max<float, WARP_SIZE>(value);
    // WRAN: need to broadcast value to all threads within warp
    value = __shfl_sync(0xffffffff, value, 0, WARP_SIZE);
    return value;
}

// naive three pass safe softmax
__global__ void softmax_kernel(const float * __restrict__ input, float * __restrict__ output, const int ncol) {
    float val;
    float max_val = - FLT_MAX;
    float exp_sum = 1e-9f;

    // max
#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
        max_val = max(input[blockIdx.x * ncol + i], max_val);
    }
    __syncthreads();

    max_val = block_reduce_max<float, BLOCK_SIZE>(max_val);

    // if (threadIdx.x == blockDim.x - 1) 
    // {
    //     printf("max_value:%.3f blockIdx.x:%d, blockIdx.y:%d\n", max_val, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
    // }

    // exp sum
#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
        exp_sum += __expf(input[blockIdx.x * ncol + i] - max_val);
    }
    __syncthreads();
    exp_sum = block_reduce_sum<float, BLOCK_SIZE>(exp_sum);

    // if (threadIdx.x == blockDim.x - 1) 
    // {
    //     printf("exp_sum:%.3f, blockIdx.x:%d, blockIdx.y:%d\n", exp_sum, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
    // }
#pragma unroll
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
        val = __expf(input[blockIdx.x * ncol + i] - max_val) / exp_sum;
        output[blockIdx.x * ncol + i] = val; 
    }
}

void launch_softmax_kernel(const float * __restrict__ input, float * __restrict__ output, const int nrow, const int ncol, cudaStream_t stream)
{
    dim3 grid(nrow);
    dim3 block(BLOCK_SIZE);
    softmax_kernel<<<grid, block, 0, stream>>>(input, output, ncol);
}

// online softmax
// naive softmax(x) = \frac{e^{x_{i}}}{\sum_{i=0}^{n} e^{x_{i}}}
// safe softmax softmax(x) = \frac{e^{x_{i} - x_{max}}}{\sum_{i=0}^{n} e^{x_{i} - x_{max}}}
// let m = x_{max}, provided a vector with 2 splits [a, b], then
// m = max(m_{a}, m_{b})
// d = d_{a} * exp(m_{a} - m) + d_{b} * exp(m_{b} - m)

struct __align__(8) MD_F
{
    float m;
    float d;
};

struct MDFOp
{
    __device__ __forceinline__ MD_F operator() (MD_F &a, MD_F &b)
    {
        MD_F ret;
        ret.m = max(a.m, b.m);
        ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
        return ret;
    }
};

__global__ void online_softmax_kernel(const float * __restrict__ input, float * __restrict__ output, const int ncol)
{
    MD_F mdf_ret, mdf_tmp;
    mdf_ret.m = - FLT_MAX;
    mdf_ret.d = 0.0f;

    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        mdf_tmp.m = input[blockIdx.x * ncol + i];
        mdf_tmp.d = 1.0f;
        mdf_ret = MDFOp()(mdf_ret, mdf_tmp);
    }
    __syncthreads();

    typedef cub::BlockReduce<MD_F, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storge;
    __shared__ MD_F mdf_total;
    mdf_ret = BlockReduce(temp_storge).Reduce(mdf_ret, MDFOp());

    if (threadIdx.x == 0)
    {
        mdf_total = mdf_ret;
    }
    __syncthreads();

    // compute softmax
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        output[blockIdx.x * ncol + i] = __expf(input[blockIdx.x * ncol + i] - mdf_total.m) / mdf_total.d;
    }
}

void launch_online_softmax_kernel(const float * __restrict__ input, float * __restrict__ output, const int nrow, const int ncol, cudaStream_t stream)
{
    dim3 grid(nrow);
    dim3 block(BLOCK_SIZE);
    online_softmax_kernel<<<grid, block, 0, stream>>>(input, output, ncol);
}