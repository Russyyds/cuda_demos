#include <stdio.h>
#include <cuda.h>
namespace {
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = 256;
}
template <const int BlockSize = BLOCK_SIZE>
__global__ void block_reduce_f32_kernel(float* input, float* output, const int n);