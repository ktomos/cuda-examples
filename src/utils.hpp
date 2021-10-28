#pragma once

#define CUDA_CHECK(call)                                             \
  {                                                                  \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
      printf("Cuda Error %d at %s:%d\n", error, __FILE__, __LINE__); \
      printf("    %s\n", cudaGetErrorString(error));                 \
    }                                                                \
  }

#define ASSERT(exp)                                                       \
  {                                                                       \
    if (!exp)                                                             \
    {                                                                     \
      printf("Assertion Error %d at %s:%d\n", error, __FILE__, __LINE__); \
      printf("    %s\n", #exp);                                           \
    }                                                                     \
  }

#define CEIL_DIV(X, Y) (((X) + (Y)-1) / (Y))
