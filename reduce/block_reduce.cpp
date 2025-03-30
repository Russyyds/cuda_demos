#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "block_reduce.cuh"

int main(int argc, char *argv[]) {
    const int N = 1024;
    const int size = N * sizeof(float);
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;
    }

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid(1);
    block_reduce_f32_kernel<<<grid, block>>>(d_in, d_out, N);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum: %f\n", *h_out);

    // Free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}