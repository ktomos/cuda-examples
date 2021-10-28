#include "gemm.hpp"
#include <memory>
#include <random>

template <typename T, class Fn>
void call_gemm(Fn gemm_func, size_t m, size_t n, size_t k, T alpha, T beta) {

  // allocate memory
  auto a = std::make_unique<T[]>(m * k);
  auto b = std::make_unique<T[]>(k * n);
  auto c = std::make_unique<T[]>(m * n);

  // init inputs
  std::default_random_engine engine(0);
  std::uniform_int_distribution<> dist(-9, 9);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      a[i * k + j] = dist(engine);
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i * n + j] = dist(engine);
    }
  }

  // execute gemm
  gemm_func(m, n, k, alpha, a.get(), b.get(), beta, c.get());
}

int main() {

  printf("%4s, %4s, %4s, %8s, %9s\n", "M", "N", "K", "GFLOPS", "sec");
  for (size_t size = 16; size <= 8192; size *= 2) {
    call_gemm<float>(gemm_cuda<float>, size, size, size, 1, 0);
  }
  return 0;
}
