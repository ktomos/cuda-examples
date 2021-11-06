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

struct GemmParam {
  static constexpr int THREAD_X = 16;
  static constexpr int THREAD_Y = 16;
  static constexpr int THREAD_XY = THREAD_X * THREAD_Y;

  // register blocking per thread
  static constexpr int MR = 8;
  static constexpr int NR = 4;

  // sheard memory blocking per grid
  static constexpr int MB = THREAD_Y * MR;
  static constexpr int NB = THREAD_X * NR;
  static constexpr int KB = 8;

  // number of memory load per thread
  static_assert((MB * KB) % (THREAD_X * THREAD_Y) == 0);
  static_assert((KB * NB) % (THREAD_X * THREAD_Y) == 0);
  static constexpr int N_LOADS_A = (MB * KB) / (THREAD_X * THREAD_Y);
  static constexpr int N_LOADS_B = (KB * NB) / (THREAD_X * THREAD_Y);
};

template <typename T, typename Param>
__global__ void gemm_kernel(const int M, const int N, const int K,
                            const T alpha, const T *A, const T *B, const T beta,
                            T *C) {
  __shared__ T shared_A[2][Param::MB][Param::KB]; // 2 x MB x KB
  __shared__ T shared_B[2][Param::KB][Param::NB]; // 2 x KB x NB

  const int n_base = blockIdx.x * Param::NB;
  const int m_base = blockIdx.y * Param::MB;
  const int tidx = threadIdx.x, tidy = threadIdx.y;

  T value[Param::MR][Param::NR] = {};
  T a_buffer[Param::N_LOADS_A], b_buffer[Param::N_LOADS_B];

  auto load_mem_to_reg = [&](int k_base) {
  // load A from device memory
#pragma unroll
    for (int i = 0; i < Param::N_LOADS_A; ++i) {
      const int index = (i * Param::THREAD_Y + tidy) * Param::THREAD_X + tidx;
      const int sheard_m_ofs = index / Param::KB;
      const int sheard_k_ofs = index % Param::KB;
      a_buffer[i] = A[(m_base + sheard_m_ofs) * K + (k_base + sheard_k_ofs)];
    }
    // load B from device memory
#pragma unroll
    for (int i = 0; i < Param::N_LOADS_B; ++i) {
      const int index = (i * Param::THREAD_Y + tidy) * Param::THREAD_X + tidx;
      const int sheard_k_ofs = index / Param::NB;
      const int sheard_n_ofs = index % Param::NB;
      b_buffer[i] = B[(k_base + sheard_k_ofs) * N + (n_base + sheard_n_ofs)];
    }
  };

  auto store_reg_to_shared_mem = [&](int shead_mem_id) {
  // store A to shared memory
#pragma unroll
    for (int i = 0; i < Param::N_LOADS_A; ++i) {
      const int index = (i * Param::THREAD_Y + tidy) * Param::THREAD_X + tidx;
      const int sheard_m_ofs = index / Param::KB;
      const int sheard_k_ofs = index % Param::KB;
      shared_A[shead_mem_id][sheard_m_ofs][sheard_k_ofs] = a_buffer[i];
    }
    // store B to shared memory
#pragma unroll
    for (int i = 0; i < Param::N_LOADS_B; ++i) {
      const int index = (i * Param::THREAD_Y + tidy) * Param::THREAD_X + tidx;
      const int sheard_k_ofs = index / Param::NB;
      const int sheard_n_ofs = index % Param::NB;
      shared_B[shead_mem_id][sheard_k_ofs][sheard_n_ofs] = b_buffer[i];
    }
  };

  auto execute_sub_matmul = [&](int shead_mem_id) {
  // execute MR X NR X KB sub matmul per thread
  // (= execute MB X NB X KB sub matmul per grid)
#pragma unroll
    for (int mr = 0; mr < Param::MR; ++mr) {
#pragma unroll
      for (int nr = 0; nr < Param::NR; ++nr) {
#pragma unroll
        for (int sheard_k_ofs = 0; sheard_k_ofs < Param::KB; ++sheard_k_ofs) {
          const int sheard_n_ofs = nr * Param::THREAD_X + tidx;
          const int sheard_m_ofs = mr * Param::THREAD_Y + tidy;
          value[mr][nr] += shared_A[shead_mem_id][sheard_m_ofs][sheard_k_ofs] *
                           shared_B[shead_mem_id][sheard_k_ofs][sheard_n_ofs];
        }
      }
    }
  };

  int k_base = 0;
  int shead_mem_id = 0;
  // prologue of main loop
  {
    load_mem_to_reg(k_base);
    store_reg_to_shared_mem(shead_mem_id);
  }
  // main loop
  for (k_base += Param::KB; k_base + Param::KB <= K; k_base += Param::KB) {
    __syncthreads();
    load_mem_to_reg(k_base);
    execute_sub_matmul(shead_mem_id);
    shead_mem_id ^= 1;
    store_reg_to_shared_mem(shead_mem_id);
  }
  // epilogue of main loop
  {
    __syncthreads();
    execute_sub_matmul(shead_mem_id);
  }

  // rest of K blocking
  for (int k_ofs = k_base; k_ofs < K; ++k_ofs) {
#pragma unroll
    for (int mr = 0; mr < Param::MR; ++mr) {
#pragma unroll
      for (int nr = 0; nr < Param::NR; ++nr) {
        const int n_ofs = n_base + nr * Param::THREAD_X + tidx;
        const int m_ofs = m_base + mr * Param::THREAD_Y + tidy;
        value[mr][nr] += A[m_ofs * K + k_ofs] * B[k_ofs * N + n_ofs];
      }
    }
  }

  // store results to device memory
#pragma unroll
  for (int mr = 0; mr < Param::MR; ++mr) {
#pragma unroll
    for (int nr = 0; nr < Param::NR; ++nr) {
      const int n_ofs = n_base + nr * Param::THREAD_X + tidx;
      const int m_ofs = m_base + mr * Param::THREAD_Y + tidy;
      if (n_ofs < N && m_ofs < M) {
        C[m_ofs * N + n_ofs] = alpha * value[mr][nr] + beta;
      }
    }
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

  const dim3 block(GemmParam::THREAD_X, GemmParam::THREAD_Y);
  const dim3 grid(CEIL_DIV(n, GemmParam::NB), CEIL_DIV(m, GemmParam::MB));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  gemm_kernel<T, GemmParam>
      <<<grid, block>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float millisec = 0;
  cudaEventElapsedTime(&millisec, start, stop);
  const double sec = double(millisec) / 1e3;
  const double gflops = double(m * n * k * 2) / sec / 1e9;
  [[maybe_unused]] const double band =
      double(m * k + k * n + m * n) * sizeof(T) / sec / 1e9;

  printf("%4ld, %4ld, %4ld, %8.2f, %9.6f\n", m, n, k, gflops, sec);

  CUDA_CHECK(cudaMemcpy(C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

template void gemm_cuda(const size_t m, const size_t n, const size_t k,
                        const float alpha, const float *A, const float *B,
                        const float beta, float *C);
