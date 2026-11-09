#!/usr/bin/env bash
# Submit from hw1-asr on the Edinburgh cluster: sbatch run_task2.sbatch
#SBATCH --job-name=mls-task2
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=task2-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo 'Submit with sbatch on the Edinburgh cluster.' >&2
    exit 1
fi
case "$(hostname -f)" in
    *.inf.ed.ac.uk) ;;
    *) echo 'This script is restricted to the Edinburgh cluster.' >&2; exit 1 ;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from hw1-asr}"
test -f csrc/task2_reduce.cu

export PATH=/opt/cuda-12.8.0/bin:$PATH
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$HOME/.conda/envs}"
conda_base="${CONDA_BASE:-/opt/conda}"
if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then conda_base="$HOME/miniconda3"; fi
source "$conda_base/etc/profile.d/conda.sh"
conda activate "${ASR_ENV:-mls-asr}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVCCFLAGS='-O3 -std=c++17 -lineinfo -diag-suppress 177'
export SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')"
test -n "$SM"
command -v compute-sanitizer >/dev/null

run_dir="$PWD/build/task2-validation/$SLURM_JOB_ID"
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
    compute-sanitizer --version
    printf 'NVCCFLAGS=%s\nSM=%s\n' "$NVCCFLAGS" "$SM"
    python --version
    python -m pip list --format=freeze
    python - <<'PY'
import importlib
import sys
from huggingface_hub import snapshot_download
print("Python executable:", sys.executable)
for name in ("torch", "transformers", "numpy", "safetensors", "huggingface_hub"):
    module = importlib.import_module(name)
    print(name, module.__version__, module.__file__)
print("Model snapshot:", snapshot_download("zai-org/GLM-ASR-Nano-2512", local_files_only=True))
PY
} > "$run_dir/environment.log" 2>&1
cat "$run_dir/environment.log"
sha256sum csrc/* tests/* Makefile run.sh run_task2.sbatch > "$run_dir/source.sha256"

run_logged build make -j "${SLURM_CPUS_PER_TASK:-2}" all "$BUILD/test_task2_helpers"
run_logged task1-regression make test1
run_logged correctness make test2
run_logged helpers "$BUILD/test_task2_helpers"
for tool in memcheck racecheck synccheck; do
    run_logged "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 \
        "$BUILD/test_task2_reduce" small
    run_logged "helpers-$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 \
        "$BUILD/test_task2_helpers"
done
run_logged model-task1 bash ./run.sh --cuda task1 --profile
run_logged model-task1-task2 bash ./run.sh --cuda task1,task2 --profile

# Reuse the supplied Makefile with isolated source copies for each experiment.
python - "$run_dir" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
def variant(name):
    dest = root / ("src-" + name)
    shutil.copytree("csrc", dest)
    return dest

def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one replacement in {path}: {old}")
    path.write_text(text.replace(old, new))

for threads in (32, 64, 128, 512, 1024):
    dest = variant(str(threads))
    replace_once(dest / "task2_reduce.cu", "constexpr int REDUCE_THREADS = 256;",
                 f"constexpr int REDUCE_THREADS = {threads};")

dest = variant("no-barrier")
replace_once(dest / "task2_reduce.cuh",
             "  __syncthreads();  // 4. a second call reuses partial[]; nobody may write it until all have read",
             "  // EXPERIMENT ONLY: final scratch-reuse barrier removed.")

dest = variant("atomic")
replace_once(dest / "task2_reduce.cu",
             "  const float inv_rms = rsqrtf(block_reduce_sum(sq) / cols + eps);",
             """  __shared__ float block_sq;
  if (threadIdx.x == 0) block_sq = 0.0f;
  __syncthreads();
  atomicAdd(&block_sq, sq);
  __syncthreads();
  const float inv_rms = rsqrtf(block_sq / cols + eps);""")
PY
for variant in 32 64 128 512 1024 atomic no-barrier; do
    variant_build="$run_dir/$variant"
    run_logged "build-$variant" make -j "${SLURM_CPUS_PER_TASK:-2}" \
        SRC="$run_dir/src-$variant" BUILD="$variant_build" "$variant_build/test_task2_reduce"
done
for round in 1 2 3; do
    variants='32 64 128 256 512 1024 atomic'
    if [[ "$round" == 2 ]]; then variants='atomic 1024 512 256 128 64 32'; fi
    for variant in $variants; do
        binary="$run_dir/$variant/test_task2_reduce"
        if [[ "$variant" == 256 ]]; then binary="$BUILD/test_task2_reduce"; fi
        run_logged "variant-$variant-run-$round" "$binary"
    done
done
for tool in memcheck racecheck synccheck; do
    run_logged "atomic-$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 \
        "$run_dir/atomic/test_task2_reduce" small
done

# The deliberately unsafe copy is never used for model inference or acceptance.
if run_logged no-barrier-test "$run_dir/no-barrier/test_task2_reduce"; then
    echo 'No-barrier ordinary test passed; this does not prove absence of races.'
else
    grep -q 'Task 2 (reductions): SOME CASES FAILED' "$run_dir/no-barrier-test.log"
fi
if run_logged no-barrier-racecheck compute-sanitizer --tool racecheck \
        --racecheck-report analysis --print-limit 30 --error-exitcode 99 \
        "$run_dir/no-barrier/test_task2_reduce" small; then
    echo 'No race was detected in this no-barrier run; retain the correct barrier.'
else
    status=$?
    if [[ "$status" != 99 ]]; then exit "$status"; fi
    echo 'Racecheck rejected the experimental no-barrier variant.'
fi
sha256sum -c "$run_dir/source.sha256" > "$run_dir/source-integrity.log"
echo 'Task 2 validation complete. Default source and correct synchronization retained.'
printf 'Logs: %s\n' "$run_dir"
