cmake_minimum_required(VERSION 3.9)
project(CudaExamples)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA 11.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

cuda_add_executable(hello_world src/helo_world.cu)


find_package(GTest REQUIRED)

include_directories(include)
cuda_add_executable(test_gemm src/gemm.cu test/test_gemm.cpp)
target_link_libraries(test_gemm GTest::GTest GTest::Main)
