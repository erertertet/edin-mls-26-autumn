# 1. A First CUDA Kernel

Slides 19 and 36–37.

## `hello.cu` (slide 19)

```cuda
__global__ void kernel() {
    printf("Block %d of %d, Thread %d of %d\n",
        blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main() {
    kernel<<<4, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```bash
make 1-first-kernel/hello && ./1-first-kernel/hello
```

`<<<4, 3>>>` launches 4 blocks × 3 threads = 12 threads, so there are 12 lines. Blocks run in any order:

```
Block 3 of 4, Thread 0 of 3
Block 3 of 4, Thread 1 of 3
Block 3 of 4, Thread 2 of 3
Block 1 of 4, Thread 0 of 3
...
```

Questions from the slide:
1. What is a block? What is a thread?
2. What does `<<<4, 3>>>` launch?
3. Why `cudaDeviceSynchronize()`?

## `qualifiers.cu` (slides 36–37)

| Qualifier | Runs on | Called from |
|-----------|---------|-------------|
| `__global__` | GPU | host, with `<<<grid, block>>>`; returns `void`; cannot recurse |
| `__device__` | GPU | device code only (may recurse) |
| `__host__` | CPU | host (the default) |
| `__host__ __device__` | both | both; `__CUDA_ARCH__` is set only in device code |

```bash
make 1-first-kernel/qualifiers && ./1-first-kernel/qualifiers
```

```
gpu hello!
gpu hello!
cpu hello!
```

- `<<<1, 2>>>` = 1 block × 2 threads → two GPU lines.
- The launch is asynchronous. `cudaDeviceSynchronize()` waits for the kernel, and that is also when the device `printf` buffer is printed, so the GPU lines come first.
- Try deleting `cudaDeviceSynchronize()`. The GPU lines may then come after the cpu line, or not be printed at all.
- Device code uses `printf`, not `std::cout`.

## 完成记录（2026-11-05）

两份示例源码已完整，无需补空。本节已在集群 GPU 节点完成编译、运行和去除同步的对照实验；未在本地执行 CUDA 编译或运行。

| 项目 | 实测记录 |
|------|----------|
| 源码版本 | `9f626d4`；本地与集群的两份 `.cu` 文件哈希一致 |
| Slurm 作业 | `3700193`，单节点、单进程、一张 GPU |
| 计算节点 | `damnii12.inf.ed.ac.uk` |
| GPU / 编译器 | NVIDIA GeForce RTX 2080 Ti / CUDA 12.8.93 |

### 运行结果

- `hello`：输出 **12 行**，恰好覆盖所有 `(blockIdx.x, threadIdx.x)` 组合：block 为 0–3，thread 为 0–2。本次 block 输出顺序为 `2 → 0 → 1 → 3`，不应依赖此顺序。
- `qualifiers`：输出如下，两个 GPU 线程各调用一次 `gpu_hello()`，同步后 CPU 调用 `cpu_hello()`。

```text
gpu hello!
gpu hello!
cpu hello!
```

临时复制 `qualifiers.cu` 并移除 `cudaDeviceSynchronize()`，使用相同编译选项运行，本次仅观察到：

```text
cpu hello!
```

这说明没有等待时，不能保证 GPU 打印在程序退出前可见；其他运行也可能观察到不同顺序。实验未改动原始源码，临时文件已删除。

### 三道问题的答案

1. **Block 和 thread 是什么？** Thread 是 kernel 的一次执行实例，有自己的 `threadIdx` 和寄存器。Block 是一组线程，块内可以共享 shared memory 并进行同步；`blockIdx` 标识该块在 grid 中的位置。
2. **`<<<4, 3>>>` 启动什么？** 一个含 4 个 block 的 grid，每个 block 有 3 个 thread，共执行 12 次 kernel。这里的全局线程编号可写为 `blockIdx.x * blockDim.x + threadIdx.x`。
3. **为什么需要 `cudaDeviceSynchronize()`？** Kernel launch 对 CPU 是异步的。同步等待此前的设备工作完成，也是刷新 GPU `printf` 输出的时点，确保本例中 GPU 输出先于后续 CPU 输出。

### 在集群复现

在 `mlp` 登录节点申请 GPU 后执行；不要在本机或登录节点编译。

```bash
srun -p Teaching --nodes=1 --ntasks=1 --nodelist=damnii12 \
  --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:05:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd ~/edin-mls-26-autumn/cuda-tutorial
make -B 1-first-kernel/hello 1-first-kernel/qualifiers
./1-first-kernel/hello
./1-first-kernel/qualifiers
exit
```

显式指定 `--ntasks=1`，避免一次启动多个示例进程；结束后用 `exit` 释放 GPU。
