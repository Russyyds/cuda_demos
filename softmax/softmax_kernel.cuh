#include <cuda_runtime.h>

void launch_softmax_kernel(const float * __restrict__ input, float * __restrict__ output, const int nrow, const int ncol, cudaStream_t stream);

void launch_online_softmax_kernel(const float * __restrict__ input, float * __restrict__ output, const int nrow, const int ncol, cudaStream_t stream);