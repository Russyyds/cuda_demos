set(SRC query_cuda_engine.cu)
add_executable(query_cuda_engine ${SRC})
install(TARGETS query_cuda_engine DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)