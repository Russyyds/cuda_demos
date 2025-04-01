#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cuda_runtime.h"
#include "block_reduce.cuh"

int main(int argc, char *argv[]) {
    const int N = 256;
    const int size = N * sizeof(float);
    const int half_size = N * sizeof(half);

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(sizeof(float));
    half *h_in_f16 = (half*)malloc(half_size);
    float *h_out_f16 = (float*)malloc(sizeof(float));
    __nv_bfloat16 *h_in_bf16 = (__nv_bfloat16*)malloc(half_size);
    float* h_out_bf16 = (float*)malloc(sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = i;
        h_in_f16[i] = __float2half(i);
        h_in_bf16[i] = __float2bfloat16(i);
    }

    float *d_in, *d_out;
    half *d_in_f16; 
    float *d_out_f16;
    __nv_bfloat16 *d_in_bf16;
    float *d_out_bf16;

    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, sizeof(float));
    cudaMalloc((void **)&d_in_f16, half_size);
    cudaMalloc((void **)&d_out_f16, sizeof(float));
    cudaMalloc((void **)&d_in_bf16, half_size);
    cudaMalloc((void **)&d_out_bf16, sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_f16, h_in_f16, half_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_bf16, h_in_bf16, half_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid(N/BLOCK_SIZE);
    launch_block_reduce_f32(d_in, d_out, N, grid, block);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("FP32 sum: %f\n", *h_out);

    launch_block_reduce_f16(d_in_f16, d_out_f16, N, grid, block);
    cudaMemcpy(h_out_f16, d_out_f16, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("FP16 sum:%f\n", *h_out_f16);

    launch_block_reduce_bf16(d_in_bf16, d_out_bf16, N, grid, block);
    cudaMemcpy(h_out_bf16, d_out_bf16, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("BF16 sum:%f\n", *h_out_bf16);

    // Free memory
    free(h_in);
    free(h_out);
    free(h_in_f16);
    free(h_out_f16);
    free(h_in_bf16);
    free(h_out_bf16);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_in_f16);
    cudaFree(d_out_f16);
    cudaFree(d_in_bf16);
    cudaFree(d_out_bf16);
    return 0;
}