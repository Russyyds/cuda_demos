# 查找当前目录下的所有源文件
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

# 添加可执行文件
add_executable(block_reduce ${SOURCES})

# 设置CUDA架构
set_target_properties(block_reduce PROPERTIES CUDA_ARCHITECTURES "80;86;89")

install(TARGETS block_reduce DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)