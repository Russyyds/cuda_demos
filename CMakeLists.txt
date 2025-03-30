cmake_minimum_required(VERSION 3.10)
project(cuda_demos LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include(CheckLanguage)
check_language(CUDA)

add_subdirectory(cuda_stream)
add_subdirectory(query_cuda_devices)
add_subdirectory(reduce)