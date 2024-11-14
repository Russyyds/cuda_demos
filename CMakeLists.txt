cmake_minimum_required(VERSION 3.4)
project(cuda_demos LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCH_BIN "80;86;89;")

add_subdirectory(cuda_stream)
add_subdirectory(query_cuda_devices)