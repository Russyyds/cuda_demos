#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>
#include "softmax_kernel.cuh"

void profile_kernel(float* d_in, float* d_out, int warmup, int round, int ncol, int nrow) {
    float elapsed_time;
    cudaEvent_t start, end;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // warm up naive softmax
    for (int i = 0; i < warmup; i++)
        launch_softmax_kernel(d_in, d_out, nrow, ncol, stream);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < round; i++) {
        launch_softmax_kernel(d_in, d_out, nrow, ncol, stream);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("naive softmax kernel time:%.6f ms\n", elapsed_time / round);

    // warm up online softmax
    for (int i = 0; i < warmup; i++)
        launch_online_softmax_kernel(d_in, d_out, nrow, ncol, stream);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < round; i++) {
        launch_online_softmax_kernel(d_in, d_out, nrow, ncol, stream);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("onlinesoftmax kernel time:%.6f ms\n", elapsed_time / round);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: softmax nrow ncol\n");
        exit(-1);
    }
    const int nrow = std::stoi(argv[1]);
    const int ncol = std::stoi(argv[2]);
    // [nrow, ncol]
    float * h_in = (float* ) malloc(nrow * ncol * sizeof(float));
    float * h_out = (float* ) malloc(nrow * ncol * sizeof(float));
    float * d_in;
    float * d_out;
    const int num_element = nrow * ncol;
    const int warmup = 5;
    const int round = 20;

    for (int i = 0; i < num_element; i ++) {
        h_in[i] = i * 1.0;
        h_out[i] = -1.0;
    }
    // printf("softmax input:\n");
    // for (int i = 0; i < num_element; i ++) {
    //     if (i % ncol == 0 && i > 0) {
    //         printf("\n");
    //     }
    //     printf("%.2f,", h_in[i]);
    // }
    // printf("\n");

    cudaMalloc((void**) &d_in, num_element * sizeof(float));
    cudaMalloc((void**) &d_out, num_element * sizeof(float));

    cudaMemcpy(d_in, h_in, num_element * sizeof(float), cudaMemcpyHostToDevice);
    profile_kernel(d_in, d_out, warmup, round, ncol, nrow);
    // profile_kernel(d_in, d_out, 0, 1, ncol, nrow);
    cudaMemcpy(h_out, d_out, num_element * sizeof(float), cudaMemcpyDeviceToHost);

    printf("softmax output:\n");
    int last = std::max(num_element - 10, 0);
    for (int i = last; i < num_element; i ++) {
        if (i % ncol == 0 && i > 0) {
            printf("\n");
        }
        printf("%.5f,", h_out[i]);
    }
    printf("\n");

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}