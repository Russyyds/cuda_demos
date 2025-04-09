#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dot_product_kernel.cuh"

int main(int argc, char** argv) {
    const int N = 1024;
    const int size = N * sizeof(float);
    const int size_f16 = N * sizeof(half);
    float* h_a = (float* )malloc(size);
    float* h_b = (float* )malloc(size);
    float* h_out = (float* )malloc(sizeof(float));
    half* h_a_f16 = (half* )malloc(size_f16);
    half* h_b_f16 = (half* )malloc(size_f16);
    float* h_out_f16 = (float* )malloc(sizeof(float));
    const int warmup = 5;
    const int round = 1000;
    float elapse_time;

    for (int i = 0; i < N; i ++) {
        h_a[i] = i;
        h_b[i] = 2;
        h_a_f16[i] = __float2half(h_a[i]);
        h_b_f16[i] = __float2half(h_b[i]);
    }

    float* d_a;
    float* d_b;
    float* d_out;
    half* d_a_f16;
    half* d_b_f16;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_out, sizeof(float));
    cudaMalloc((void**)&d_a_f16, size_f16);
    cudaMalloc((void**)&d_b_f16, size_f16);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_f16, h_a_f16, size_f16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_f16, h_b_f16, size_f16, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(N/BLOCK_SIZE);
    dim3 grid4(N/BLOCK_SIZE/4);
    // warmup
    for (int i = 0; i < warmup; i++) {
        launch_dot_prod_f32(d_a, d_b, d_out, N, grid, block);
    }

    // profile fp32 kernel
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < round; i++) {
        launch_dot_prod_f32(d_a, d_b, d_out, N, grid, block);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse_time, start, end);
    elapse_time /= round;
    cudaMemset(d_out, 0, sizeof(float));
    launch_dot_prod_f32(d_a, d_b, d_out, N, grid, block);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("FP32 dot_prod:%f time:%f ms\n", *h_out, elapse_time);

    // profile fp32x4 kernel
    cudaEventRecord(start);
    for (int i = 0; i < round; i++) {
        launch_dot_prod_f32x4(d_a, d_b, d_out, N, grid4, block);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse_time, start, end);
    elapse_time /= round;

    cudaMemset(d_out, 0, sizeof(float));
    launch_dot_prod_f32x4(d_a, d_b, d_out, N, grid4, block);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("FP32x4 dot_prod:%f time:%f ms\n", *h_out , elapse_time);

    cudaMemset(d_out, 0, sizeof(float));
    cudaEventRecord(start);
    for (int i = 0; i < round; i ++) {
        launch_dot_prod_f16(d_a_f16, d_b_f16, d_out, N, grid, block);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse_time, start, end);
    elapse_time /= round;
    cudaMemset(d_out, 0, sizeof(float));
    launch_dot_prod_f16(d_a_f16, d_b_f16, d_out, N, grid, block);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP16 dot_prod:%f time:%f\n", *h_out, elapse_time);

    return 0;
}
