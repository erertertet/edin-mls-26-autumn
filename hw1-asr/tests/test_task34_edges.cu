// Reuse the supplied CPU references without changing the official suites.
#define main supplied_gemm_main
#include "test_task3_gemm.cu"
#undef main
#define main supplied_attention_main
#include "test_task4_attention.cu"
#undef main

int main() {
  g_small_only = true;
  print_gpu();
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  bool ok = true;
  const GemmCase gemm[] = {
      {"GELU without bias, tile boundary", 17, 15, 31, false, 1},
      {"exact output tiles, partial K", 16, 32, 17, true, 0},
  };
  for (const auto& t : gemm) ok = run_case(t, handle) && ok;
  CHECK_CUBLAS(cublasDestroy(handle));
  const AttnCase attention[] = {
      {"96 dimensions, GQA, spare warp", 1, 4, 1, 3, 65, 96, false, 4.0f},
      {"96 dimensions, shifted causal", 1, 2, 1, 5, 37, 96, true, 4.0f},
      {"single key, batched GQA", 2, 4, 2, 1, 1, 64, true, 1.0f},
  };
  for (const auto& t : attention) ok = run_case(t) && ok;
  return finish("Task 3/4 boundary cases", ok);
}
