#include "gemm.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <random>

using GemmParams = std::tuple<int,    // m
                              int,    // n
                              int,    // k
                              float,  // alpha
                              float>; // beta

class GemmTest : public testing::TestWithParam<GemmParams> {
public:
  void runtest(GemmParams param) {
    const auto [m, n, k, alpha, beta] = param;

    auto a = std::make_unique<float[]>(m * k);
    auto b = std::make_unique<float[]>(k * n);
    auto c_expect = std::make_unique<float[]>(m * n);
    auto c_actual = std::make_unique<float[]>(m * n);

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
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        c_expect[i * n + j] = dist(engine);
        c_actual[i * n + j] = dist(engine);
      }
    }

    gemm_cpu<float>(m, n, k, alpha, a.get(), b.get(), beta, c_expect.get());
    gemm_cuda<float>(m, n, k, alpha, a.get(), b.get(), beta, c_actual.get());

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        const auto expect = c_expect[i * n + j];
        const auto actual = c_actual[i * n + j];
        ASSERT_NEAR(expect, actual, 1e-4f);
      }
    }
  }
};

TEST_P(GemmTest, ParamTest) { runtest(GetParam()); }

using testing::Combine;
using testing::Values;

INSTANTIATE_TEST_CASE_P(GemmMNKBasicTest, GemmTest,
                        Combine(Values(16, 64), // m
                                Values(16, 64), // n
                                Values(16, 64), // k
                                Values(1.0),    // alpha
                                Values(1.0)));  // beta

INSTANTIATE_TEST_CASE_P(DISABLED_GemmMNKSmallTest, GemmTest,
                        Combine(Values(1, 2, 3), // m
                                Values(1, 2, 3), // n
                                Values(1, 2, 3), // k
                                Values(1.0),     // alpha
                                Values(1.0)));   // beta

INSTANTIATE_TEST_CASE_P(DISABLED_GemmMNKEdgeTest, GemmTest,
                        Combine(Values(61, 67), // m
                                Values(61, 67), // n
                                Values(61, 67), // k
                                Values(1.0),    // alpha
                                Values(1.0)));  // beta

INSTANTIATE_TEST_CASE_P(GemmMNKLargeTest, GemmTest,
                        Values(GemmParams(128, 128, 128, 1.0, 1.0),
                               GemmParams(256, 256, 256, 1.0, 1.0),
                               GemmParams(512, 512, 512, 1.0, 1.0),
                               GemmParams(500, 500, 500, 1.0, 1.0)));

INSTANTIATE_TEST_CASE_P(GemmAlphaBetaTest, GemmTest,
                        Combine(Values(16),              // m
                                Values(16),              // n
                                Values(16),              // k
                                Values(0.0, 0.7, 1.0),   // alpha
                                Values(0.0, 0.7, 1.0))); // beta
