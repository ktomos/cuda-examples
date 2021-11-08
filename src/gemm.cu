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
  static_assert(Param::KB % 4 == 0);
  static_assert(std::is_same<T, float>::value);
  __shared__ float4 shared_A[2][Param::MB][Param::KB / 4]; // 2 x MB x KB
  __shared__ T shared_B[2][Param::KB][Param::NB];          // 2 x KB x NB

  const int n_base = blockIdx.x * Param::NB;
  const int m_base = blockIdx.y * Param::MB;
  const int tidx = threadIdx.x, tidy = threadIdx.y;
  const int tid = tidy * Param::THREAD_X + tidx;

  T value[Param::MR][Param::NR] = {};

  static_assert(Param::N_LOADS_A == 4);
  float4 a_buffer;
  float b_buffer[Param::N_LOADS_B];

  const float *A_ofs, *B_ofs[Param::N_LOADS_B];

  // init device memory load address of A
  {
    const int index = tid * 4;
    const int m_ofs = m_base + index / Param::KB;
    const int k_ofs = 0 + index % Param::KB;
    A_ofs = A + (m_ofs)*K + k_ofs;
  }
// init device memory load address of B
#pragma unroll
  for (int i = 0; i < Param::N_LOADS_B; ++i) {
    const int index = i * Param::THREAD_XY + tid;
    const int k_ofs = 0 + index / Param::NB;
    const int n_ofs = n_base + index % Param::NB;
    B_ofs[i] = B + (k_ofs)*N + n_ofs;
  }

  auto load_mem_to_reg = [&](int k_base) {
    // load A from device memory
    {
      // load by 128-bit instruction
      asm("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=f"(a_buffer.x), "=f"(a_buffer.y), "=f"(a_buffer.z),
            "=f"(a_buffer.w)
          : "l"(A_ofs)
          : "memory");
    }
    // load B from device memory
#pragma unroll
    for (int i = 0; i < Param::N_LOADS_B; ++i) {
      b_buffer[i] = *B_ofs[i];
    }
    // update device memory address of A
    A_ofs += Param::KB;
    // update device memory address of B
#pragma unroll
    for (int i = 0; i < Param::N_LOADS_B; ++i) {
      B_ofs[i] += Param::KB * N;
    }
  };

  auto store_reg_to_shared_mem = [&](int shead_mem_id) {
    // store A to shared memory
    {
      const int index = tid * 4;
      const int sheard_m_ofs = index / Param::KB;
      const int sheard_k_ofs = index % Param::KB;
      shared_A[shead_mem_id][sheard_m_ofs][sheard_k_ofs / 4] = a_buffer;
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
      const int sheard_m_ofs = mr * Param::THREAD_Y + tidy;
#pragma unroll
      for (int nr = 0; nr < Param::NR; ++nr) {
        const int sheard_n_ofs = nr * Param::THREAD_X + tidx;
#pragma unroll
        for (int k = 0; k < Param::KB / 4; ++k) {
          float4 a_buf = shared_A[shead_mem_id][sheard_m_ofs][k];
          value[mr][nr] +=
              a_buf.x * shared_B[shead_mem_id][k * 4 + 0][sheard_n_ofs];
          value[mr][nr] +=
              a_buf.y * shared_B[shead_mem_id][k * 4 + 1][sheard_n_ofs];
          value[mr][nr] +=
              a_buf.z * shared_B[shead_mem_id][k * 4 + 2][sheard_n_ofs];
          value[mr][nr] +=
              a_buf.w * shared_B[shead_mem_id][k * 4 + 3][sheard_n_ofs];
        }
      }
    }
  };

  int k_base = 0;
  // prologue of main loop
  {
    load_mem_to_reg(k_base);
    store_reg_to_shared_mem(0);
  }
  // main loop
  for (k_base += Param::KB; k_base < K - Param::KB * 2;
       k_base += Param::KB * 2) {
    __syncthreads();
    load_mem_to_reg(k_base);
    execute_sub_matmul(0);
    store_reg_to_shared_mem(1);
    __syncthreads();
    load_mem_to_reg(k_base + Param::KB);
    execute_sub_matmul(1);
    store_reg_to_shared_mem(0);
  }
  // epilogue of main loop
  int shead_mem_id = 0;
  if (k_base < K - Param::KB) {
    __syncthreads();
    load_mem_to_reg(k_base);
    execute_sub_matmul(shead_mem_id);
    shead_mem_id ^= 1;
    store_reg_to_shared_mem(shead_mem_id);
    k_base += Param::KB;
  }
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
  if (k % 4 == 0 && n % 4 == 0) {
    gemm_kernel<T, GemmParam>
        <<<grid, block>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
  }
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
