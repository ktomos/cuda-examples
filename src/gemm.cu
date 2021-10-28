#include "utils.hpp"
#include <iostream>

template <typename T>
void gemm_cpu(const size_t m, const size_t n, const size_t k, const T alpha,
              const T *A, const T *B, const T beta, T *C) {
  for (size_t im = 0; im < m; ++im) {
    for (size_t in = 0; in < n; ++in) {
      T value = 0;
      for (size_t ik = 0; ik < k; ++ik) {
        value += A[im * k + ik] * B[ik * n + in];
      }
      C[im * n + in] = alpha * value + beta;
    }
  }
}

template void gemm_cpu(const size_t m, const size_t n, const size_t k,
                       const float alpha, const float *A, const float *B,
                       const float beta, float *C);

template <typename T>
__global__ void gemm_kernel(const size_t m, const size_t n, const size_t k,
                            const T alpha, const T *A, const T *B, const T beta,
                            T *C) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int idy = blockDim.y * blockIdx.y + threadIdx.y;
  if (idx < n && idy < m) {
    T value = 0;
    for (size_t i = 0; i < k; ++i) {
      value += A[idy * k + i] * B[i * n + idx];
    }
    C[idy * n + idx] = alpha * value + beta;
  }
}

template <typename T>
void gemm_cuda(const size_t m, const size_t n, const size_t k, const T alpha,
               const T *A, const T *B, const T beta, T *C) {
  T *d_A, *d_B, *d_C;

  CUDA_CHECK(cudaMalloc((T **)&d_A, m * k * sizeof(T)));
  CUDA_CHECK(cudaMalloc((T **)&d_B, k * n * sizeof(T)));
  CUDA_CHECK(cudaMalloc((T **)&d_C, m * n * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_A, A, m * k * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, k * n * sizeof(T), cudaMemcpyHostToDevice));

  const dim3 block(16, 16, 1);
  const dim3 grid(CEIL_DIV(n, 16), CEIL_DIV(m, 16));

  gemm_kernel<<<grid, block>>>(m, n, k, alpha, d_A, d_B, beta, d_C);

  // cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

template void gemm_cuda(const size_t m, const size_t n, const size_t k,
                        const float alpha, const float *A, const float *B,
                        const float beta, float *C);
