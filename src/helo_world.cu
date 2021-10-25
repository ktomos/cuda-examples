#include <stdio.h>

__global__ void kernel_print_hello() { printf("Hello from CUDA\n"); }

int main() {
  printf("Hello from CPU\n");
  kernel_print_hello<<<1, 16>>>();
  cudaDeviceReset();
  return 0;
}
