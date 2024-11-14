//功能 
// stream1执行kernel1
// stream2等待kernel1执行完成并开始执行kernel2
// stream1执行完kernel1后继续执行kernel3
// stream1: kernel1....kernel3 
// stream2:           kernel2
//
#include <iostream>
#include <cuda_runtime.h>

// Kernel functions to perform computation
__global__ void kernel1(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] += 1;
    }
}

__global__ void kernel2(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] += 2;
    }
}

__global__ void kernel3(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] -= 1;
    }
}

int main() {
    const int data_size = 2048;
    const int print_size = 20;
    int64_t *host_data = new int64_t[data_size];
    int64_t *device_data1, *device_data2;
    for (int i = 0; i < data_size; i++) {
        host_data[i] = i;
    }

    cudaMalloc((void**)&device_data1, data_size * sizeof(int64_t));
    cudaMalloc((void**)&device_data2, data_size * sizeof(int64_t));

    cudaMemcpy(host_data, device_data1, data_size * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(host_data, device_data2, data_size * sizeof(int64_t), cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x);

    cudaStream_t stream1, stream2;
    cudaEvent_t event1;
    int high_priority, low_priority;
    cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);
    cudaStreamCreate(&stream1);
    cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, high_priority);

    cudaEventCreate(&event1);
    const int repeat = 1000;
    // 在stream1执行kernel1
    kernel1<<<gridDim, blockDim, 0, stream2>>>(device_data1, repeat);
    cudaEventRecord(event1, stream1);
    cudaStreamWaitEvent(stream2, event1);
    // 在stream2执行kernel2
    kernel2<<<gridDim, blockDim, 0, stream2>>>(device_data1, repeat);
    
    kernel3<<<gridDim, blockDim, 0, stream1>>>(device_data2, repeat);

    //同步两个stream
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaMemcpy(host_data, device_data1, data_size * sizeof(int64_t), cudaMemcpyDeviceToHost);

    std::cout << "Data after kernel1 and kernel2:" << std::endl;
    for (int i = 0; i < print_size; i++) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;

    cudaMemcpy(host_data, device_data2, data_size * sizeof(int64_t), cudaMemcpyDeviceToHost);

    std::cout << "Data after kernel3:" << std::endl;
    for (int i = 0; i < print_size; i++) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
    cudaFree(device_data1);
    cudaFree(device_data2);
    delete [] host_data;
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event1);
    return 0;
}