#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dot_product_kernel.cuh"

int main(int argc, char** argv) {
    const int N = 1024;
    const int size = N * sizeof(float);
    float* h_a = (float* )malloc(size);
    float* h_b = (float* )malloc(size);
    float* h_out = (float* )malloc(sizeof(float));

    for (int i = 0; i < N; i ++) {
        h_a[i] = i;
        h_b[i] = 2;
    }
    float* d_a;
    float* d_b;
    float* d_out;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_out, sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE);
    dim3 grid(N/BLOCK_SIZE);
    launch_dot_prod_f32(d_a, d_b, d_out, N, grid, block);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("FP32 dot_prod:%f\n", *h_out);

    return 0;
}
