# 2. Example 0: Vector Add

Slides 38–55 (and 85 for the batch job). One thread per element, `c[idx] = a[idx] + b[idx]`.

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one global index per thread
    if (idx < n) {                                     // spare threads skip the work
        c[idx] = a[idx] + b[idx];
    }
}
// gridSize × blockSize ≥ n threads
vector_add<<<gridSize, blockSize>>>(a, b, c, n);
```

## Files

| File | Slide | Run | Expected |
|------|-------|-----|----------|
| `host_path.cu` | 51 | `./host_path` | `3.000000` ten times |
| `pointer_mistake1.cu` | 47 | `./pointer_mistake1` | `CUDA error at pointer_mistake1.cu:21 code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()"` |
| `pointer_mistake2.cu` | 48 | `./pointer_mistake2` | `Segmentation fault (core dumped)` |
| `memcpy.cu` | 49 | `./memcpy` | `55` |
| `managed.cu` | 50 | `./managed` | `55` |
| `vector_add.cu` | 53 | `./vector_add 1003 256` | `PASS vector add` |
| `smoke.sbatch` | 85 | `sbatch smoke.sbatch` | job `COMPLETED`, `PASS` twice in `smoke-JOB_ID.out` |

Build everything with `make` in `cuda-tutorial/`, then run from this directory.

## The complete host path (`host_path.cu`, slides 38 and 51)

1. Allocate host and device buffers (`malloc`, `cudaMalloc`).
2. Copy inputs to the GPU (`cudaMemcpy ... cudaMemcpyHostToDevice`).
3. Launch ⌈n/256⌉ blocks of 256 threads.
4. Copy the results back and check them.
5. Free both sides.

The program is the slide code. As the slide notes say, real code should also check the launch, and the `cudaDeviceSynchronize()` after the blocking `cudaMemcpy` is redundant. `vector_add.cu` does both properly.

## Host and device memory (slides 45–50)

- `pointer_mistake1.cu`: a `malloc` pointer handed to a kernel. The kernel faults; `checkCudaErrors` reports error 700. Without the check, the program silently prints 0, and a wrong answer is worse than a crash. On systems with HMM or ATS (`nvidia-smi -q | grep "Addressing Mode"`), the GPU can read `malloc` memory and the program prints 55. The teaching cluster GPUs report `None`.
- `pointer_mistake2.cu`: `cudaMalloc` fills in a device address. Dereferencing it on the host crashes.
- `memcpy.cu`: explicit copies. `cudaMemcpy` waits for the kernel in the same stream, so no extra sync is needed.
- `managed.cu`: one pointer for both sides. Pages migrate on demand, at a hidden time cost; the host must `cudaDeviceSynchronize()` before reading.

The kernel adds `arr[1]`, `arr[2]`, ... until it reads a 0, so `a[11]` must be 0. `malloc` and `cudaMallocManaged` do not zero memory; the files use `calloc` or set `a[11] = 0`.

## Student checkpoint (`vector_add.cu`, slide 53)

On the GPU node, in this directory:

```bash
nvcc -O2 -std=c++17 -arch=native \
  vector_add.cu -o vector_add
./vector_add 1003 256
./vector_add 257 128
```

Expected output (the first line names your GPU):

```
GPU=NVIDIA GeForce RTX 2080 Ti CC=7.5 SMs=68 warp=32
n=1003 block=256 grid=4
checked=1003 failures=0
PASS vector add
```

Then:
1. Record the node and GPU.
2. Explain the bounds check: 1003 and 257 are not multiples of the block size.
3. Exit the GPU shell.

Why do both configurations produce the same answer?

### 实测结果与问题解答（2026-11-05）

#### 1. 记录计算节点和 GPU

本次在集群 `damnii12.inf.ed.ac.uk` 上复测，复用已有 Slurm 作业 `3700198` 的 GPU allocation，以单进程 step `1` 执行。所有 CUDA 编译和运行均在该 GPU 节点完成。

- GPU：NVIDIA GeForce RTX 2080 Ti。
- `CC=7.5`：计算能力版本，对应编译目标 `sm_75`。
- `SMs=68`：GPU 有 68 个流式多处理器（Streaming Multiprocessors）。
- `warp=32`：线程按每组 32 个的 warp 组织执行。
- 编译器：CUDA 12.8.93，选项为 `-O2 -std=c++17 -arch=native`。

| n | block | grid | 启动线程数 | 超出数组范围的线程数 | failures |
|---:|---:|---:|---:|---:|---:|
| 1003 | 256 | 4 | 1024 | 21 | 0 |
| 257 | 128 | 3 | 384 | 127 | 0 |
| 1003 | 128 | 8 | 1024 | 21 | 0 |

三次运行均输出 `PASS vector add`。`checked=n` 表示检查了全部 n 个输出；程序逐元素与 CPU 计算的 `a[i] + b[i]` 比较，误差大于 `1e-6` 时计为失败。`failures=0` 表示本次检查未发现不匹配。

#### 2. 为什么需要边界检查？

每个线程使用全局索引定位一个元素：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

block 数向上取整，代码为 `grid = (n + block - 1) / block`，保证线程总数足以覆盖全部元素。由于 n 不一定是 block 大小的整数倍，最后一个 block 可能只有部分线程对应真实元素。

- `n=1003, block=256`：前三个 block 处理 768 个元素，最后一个处理剩余 235 个；其余 21 个线程不执行数组访问。
- `n=257, block=128`：前两个 block 处理 256 个元素，第三个只需处理 1 个；其余 127 个线程不执行数组访问。
- `n=1003, block=128`：前七个 block 处理 896 个元素，最后一个处理 107 个；其余 21 个线程跳过数组访问。

`if (i < n)` 让有效线程执行加法，并阻止多余线程读写数组之外的地址。这两种非整除长度都通过了检查，说明本次测试覆盖了最后一个 block 的边界情况。

#### 3. 退出 GPU shell

结束 GPU 使用后，在交互式 GPU 终端执行 `exit`，退出由 `srun --pty bash` 创建的 shell，释放该 allocation。

本次复测只结束了自己的 step，保留了用户原有的交互式 GPU shell；因此不能将“退出 GPU shell”记为本次已完成。

#### “Same answer”具体指什么？

原来的两组参数分别使用 n=1003 和 n=257，输出数组长度不同；两个 `PASS` 只说明它们各自正确，不能证明两个完整数组相同。

为验证 block 大小是否影响结果，额外固定 n=1003，比较 block=256 与 block=128。输入由索引 i 确定，因此两次使用相同的输入数组。通过临时源码副本打印完整输出后，逐项对比得到：

```text
Full-array comparison: n=1003, block=256 vs 128, elements=1003, identical=True
```

这次实测确认：相同输入下，两种 block 配置的全部 1003 个输出元素一致。改变 block 大小只改变线程的分组和 grid 大小；每个有效索引 i 仍唯一对应一个线程，并执行相同的 `c[i] = a[i] + b[i]`，各线程之间没有结果依赖。

临时副本仅增加结果打印，原始源码未改动，临时文件已清理。本节验证的是正确性；这些小规模输入和 `PASS` 信息不能用于判断哪种配置更快。

## Batch job (`smoke.sbatch`, slide 85)

From the head node, in this directory:

```bash
sbatch smoke.sbatch
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode
cat smoke-JOB_ID.out
```

Success means `State COMPLETED`, `ExitCode 0:0` and `PASS` in the output file. "Submitted" alone is not success. `scancel JOB_ID` cancels a job you no longer need.
