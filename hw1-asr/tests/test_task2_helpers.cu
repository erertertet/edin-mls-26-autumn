// Extra contract checks for the helpers reused by later tasks.
// Build with the existing Makefile target build/sm_XX/test_task2_helpers.
#include "test_util.h"
#include "task2_reduce.cuh"

__global__ void check_reductions(float* out) {
  const int t = threadIdx.x;
  out[t] = block_reduce_sum(float(t + 1));
  out[blockDim.x + t] = block_reduce_sum(1.0f);  // reuse shared scratch
  out[2 * blockDim.x + t] = block_reduce_max(-float(t + 1));
  out[3 * blockDim.x + t] = block_reduce_max(-float(t + 7));
}

__global__ void check_summaries(float* out) {
  const MaxSum empty = {-INFINITY, 0.0f};
  const MaxSum single = {-175.0f, 1.0f};
  const MaxSum values[] = {
      maxsum_merge(empty, empty), maxsum_merge(empty, single),
      maxsum_merge(single, empty), maxsum_merge({1.0f, 1.0f}, {3.0f, 1.0f}),
      maxsum_merge(maxsum_merge({1.0f, 1.0f}, {3.0f, 1.0f}), {2.0f, 1.0f}),
      maxsum_merge({1.0f, 1.0f}, maxsum_merge({3.0f, 1.0f}, {2.0f, 1.0f})),
      maxsum_merge({-100.0f, 1.0f}, {100.0f, 1.0f}),
  };
  for (int i = 0; i < 7; ++i) {
    out[2 * i] = values[i].m;
    out[2 * i + 1] = values[i].l;
  }
}

int main() {
  print_gpu();
  bool ok = true;
  for (int threads : {32, 64, 128, 256, 512, 1024}) {
    DeviceArray output(4 * threads);
    check_reductions<<<1, threads>>>(output.ptr);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<double> expected(4 * threads);
    for (int t = 0; t < threads; ++t) {
      expected[t] = double(threads) * (threads + 1) / 2;
      expected[threads + t] = threads;
      expected[2 * threads + t] = -1;
      expected[3 * threads + t] = -7;
    }
    auto c = compare(output.download(), expected, 0, 0);
    ok = report("helper reductions, " + std::to_string(threads) + " threads", c, 0, "") && ok;
  }

  DeviceArray output(14);
  check_summaries<<<1, 1>>>(output.ptr);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  auto got = output.download();
  const bool empty_ok = std::isinf(got[0]) && got[0] < 0 && got[1] == 0;
  std::printf("%-36s %s\n", "MaxSum empty + empty", empty_ok ? "PASS" : "FAIL");
  const double pair_sum = 1 + std::exp(-2.0);
  const double triple_sum = pair_sum + std::exp(-1.0);
  std::vector<double> expected = {-175, 1, -175, 1, 3, pair_sum,
                                  3, triple_sum, 3, triple_sum, 100, 1};
  std::vector<float> finite(got.begin() + 2, got.end());
  auto c = compare(finite, expected, 1e-6, 1e-6);
  ok = report("MaxSum identities and stable merges", c, 0, "") && empty_ok && ok;
  return finish("Task 2 helpers", ok);
}
