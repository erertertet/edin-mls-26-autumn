#!/usr/bin/env bash
# From hw1-asr on the Edinburgh cluster: sbatch run_task1.sbatch
# Requires setup_env.sh to have finished on the login node.
#SBATCH --job-name=mls-task1
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=task1-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Run with sbatch on the Edinburgh cluster, inside a GPU allocation." >&2
    exit 1
fi
case "$(hostname -f)" in
    *.inf.ed.ac.uk) ;;
    *) echo "This script is restricted to the Edinburgh cluster." >&2; exit 1 ;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from hw1-asr}"
test -f csrc/task1_rope.cu

export PATH=/opt/cuda-12.8.0/bin:$PATH
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$HOME/.conda/envs}"
conda_base="${CONDA_BASE:-/opt/conda}"
if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    conda_base="$HOME/miniconda3"
fi
source "$conda_base/etc/profile.d/conda.sh"
conda activate "${ASR_ENV:-mls-asr}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVCCFLAGS='-O3 -std=c++17 -lineinfo -diag-suppress 177'
export SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')"
test -n "$SM"

run_dir="$PWD/build/task1-validation/$SLURM_JOB_ID"
mkdir -p "$run_dir"
export BUILD="$run_dir/default"
export ASR_LIB="$BUILD/libasr_kernels.so"
run_logged() {
    local name="$1"
    shift
    printf '\n>>> %s\n' "$name"
    "$@" 2>&1 | tee "$run_dir/$name.log"
}
{
    date -u --iso-8601=seconds
    hostname -f
    printf 'Slurm job: %s\n' "$SLURM_JOB_ID"
    git rev-parse HEAD
    git status --short
    nvidia-smi -L
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    nvcc --version
    printf 'NVCCFLAGS=%s\nSM=%s\n' "$NVCCFLAGS" "$SM"
    python --version
    python -m pip list --format=freeze
    python - <<'PY'
import importlib
import inspect
import sys
from huggingface_hub import snapshot_download

print("Python executable:", sys.executable)
for name in ("torch", "transformers", "numpy", "safetensors", "huggingface_hub"):
    module = importlib.import_module(name)
    print(name, module.__version__, module.__file__)
print("snapshot_download API:", inspect.signature(snapshot_download))
print("Model snapshot:", snapshot_download("zai-org/GLM-ASR-Nano-2512", local_files_only=True))
PY
} > "$run_dir/environment.log" 2>&1
cat "$run_dir/environment.log"
sha256sum csrc/* tests/* Makefile run.sh run_task1.sbatch > "$run_dir/source.sha256"

run_logged build make -j "${SLURM_CPUS_PER_TASK:-2}"
run_logged correctness make test1
if command -v compute-sanitizer >/dev/null; then
    run_logged memcheck compute-sanitizer --tool memcheck --error-exitcode 1 \
        "$BUILD/test_task1_rope" small
else
    echo "SKIPPED: compute-sanitizer is unavailable." | tee "$run_dir/memcheck.log"
fi
run_logged model-reference bash ./run.sh --cuda none
run_logged model-task1 bash ./run.sh --cuda task1 --profile

# Isolate every experiment so flags and binaries cannot contaminate the baseline.
for threads in 32 64 128 512 1024; do
    variant_src="$run_dir/src-$threads"
    variant_build="$run_dir/threads-$threads"
    mkdir -p "$variant_src"
    cp csrc/* "$variant_src/"
    sed "s/constexpr int ROPE_THREADS = 256;/constexpr int ROPE_THREADS = $threads;/" \
        csrc/task1_rope.cu > "$variant_src/task1_rope.cu"
    run_logged "build-$threads" make -j "${SLURM_CPUS_PER_TASK:-2}" \
        SRC="$variant_src" BUILD="$variant_build" "$variant_build/test_task1_rope"
done
for round in 1 2 3; do
    sizes='32 64 128 256 512 1024'
    if [[ "$round" == 2 ]]; then sizes='1024 512 256 128 64 32'; fi
    for threads in $sizes; do
        binary="$run_dir/threads-$threads/test_task1_rope"
        if [[ "$threads" == 256 ]]; then binary="$BUILD/test_task1_rope"; fi
        run_logged "threads-$threads-run-$round" "$binary"
    done
done

fast_build="$run_dir/fast-math"
run_logged build-fast-math make -j "${SLURM_CPUS_PER_TASK:-2}" BUILD="$fast_build" \
    NVCCFLAGS='-O3 -std=c++17 --use_fast_math' "$fast_build/test_task1_rope"
if run_logged fast-math "$fast_build/test_task1_rope"; then
    echo 'Fast-math experiment: all cases passed on this GPU.'
else
    # A numerical comparison failure is an experimental result, not acceptance.
    grep -q 'Task 1 (RoPE): SOME CASES FAILED' "$run_dir/fast-math.log"
    echo 'Fast-math experiment failed accuracy checks; baseline flags are retained.'
fi
sha256sum -c "$run_dir/source.sha256" > "$run_dir/source-integrity.log"
echo 'Task 1 validation complete. Default source and standard-math build retained.'
printf 'Logs: %s\n' "$run_dir"
