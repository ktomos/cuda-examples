#include "utils.hpp"

__global__ void cuda_warmup_kernel() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  b += a + tid;
}

int cuda_warmup() {
  const dim3 block(256);
  const dim3 grid(1024 * 1024 * 1024);
  for (int i = 0; i < 2; ++i) {
    cuda_warmup_kernel<<<grid, block>>>();
  }
  return 0;
}
