# Repository Guidelines

## Project Structure & Module Organization
- `cuda-tutorial/`: introductory exercises.
- `hw1-asr/`: CUDA tasks in `csrc/`, reference tests in `tests/`, guides in `docs/`, and model integration in `glm_asr_cuda/`.
- `rag/`: pipeline in `src/`, commands in `scripts/`, adapter tests in `tests/`, assets in `data/` and `models/`.
- `report/`: LaTeX template and PDF checker. `course-materials/`: collected reading resources.

## Build, Test, and Development Commands
Run these on the cluster, from the remote repository root unless noted:
- `make -C cuda-tutorial`: build lecture examples.
- `make -C hw1-asr`: build the kernel library and CUDA tests; CMake is unnecessary.
- `make -C hw1-asr test1`: run RoPE tests; use `test2` through `test4` for subsequent tasks, or `test` for all.
- Inside `hw1-asr/`, `./run.sh --cuda task1` checks model integration; `./run.sh --profile` profiles all kernels.
- `python -m unittest discover -s rag/tests -v`: run GPU-independent adapter tests.
- `python rag/scripts/check_setup.py`: validate RAG configuration without ELM requests.

Follow module READMEs for separate ASR/RAG environments and CUDA PATH setup.

## Execution Policy: Cluster Only
Run coursework programs, CUDA compilation, tests, inference, and benchmarks on the Edinburgh cluster in a Slurm allocation. Use the head node only for login, synchronization, and environment setup.

Never use this computer's GPU, including its RTX 4090, through Windows, WSL, containers, or automatic framework device selection. Local work is limited to editing and lightweight CPU-only inspection. If cluster access or allocation fails, report the blocker; do not fall back to local execution.

## Coding Style & Naming Conventions
Follow surrounding code: two-space indentation in coursework CUDA, four spaces in Python, `snake_case` functions, uppercase constants, and `taskN_operation` filenames. No repository-wide formatter/linter is configured.

## Testing & Coursework Requirements
Preserve supplied tests and interfaces. Implement task kernels yourself; do not use cuBLAS, cuDNN, CUTLASS, Thrust, or PyTorch for task computation. Verify reference tests and transcription; record GPU, toolkit, commands, and results. State skipped checks. Follow current READMEs and task guides over stale checklists.

## Commit & Pull Request Guidelines
Use imperative subjects (`Add`, `Update`, `Fix`, `Document`), matching history; Conventional Commit prefixes are optional. Keep changes focused. PRs should describe the problem, implementation, validation, hardware-dependent results, and relevant issues.

## Configuration & Agent Workflow
Keep credentials in local `.env` files; exclude secrets and generated artifacts from commits. Preserve existing edits. Confirm the specific coursework task before implementation and respect paused GPU workloads. Verify dependencies; consult official resources and reuse APIs within coursework constraints.
