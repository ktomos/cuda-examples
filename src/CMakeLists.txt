find_package(CUDA 11.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

cuda_add_executable(hello_world helo_world.cu)
