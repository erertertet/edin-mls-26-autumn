#!/usr/bin/env bash
# Submit from hw1-asr on the Edinburgh cluster: sbatch run_task34.sbatch
#SBATCH --job-name=mls-task34
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=task34-%j.out

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
test -f csrc/task3_gemm.cu
test -f csrc/task4_attention.cu

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

run_dir="$PWD/build/task34-validation/$SLURM_JOB_ID"
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
    python -m pip check
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
sha256sum csrc/* tests/* Makefile run.sh run_task34.sbatch > "$run_dir/source.sha256"

run_logged build make -j "${SLURM_CPUS_PER_TASK:-2}" all "$BUILD/test_task2_helpers" "$BUILD/test_task34_edges"
run_logged correctness-all make test
run_logged helpers "$BUILD/test_task2_helpers"
run_logged edges "$BUILD/test_task34_edges"
for task in task3_gemm task4_attention; do
    for tool in memcheck racecheck synccheck; do
        run_logged "$task-$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 \
            "$BUILD/test_$task" small
    done
done
for tool in memcheck racecheck synccheck; do
    run_logged "edges-$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 \
        "$BUILD/test_task34_edges" small
done
run_logged model-none bash ./run.sh --cuda none --profile
run_logged model-task12 bash ./run.sh --cuda task1,task2 --profile
run_logged model-task123 bash ./run.sh --cuda task1,task2,task3 --profile
run_logged model-all bash ./run.sh --cuda all --profile

# Reuse the course Makefile and reference tests with isolated source copies.
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

for tile in (8, 32):
    dest = variant(f"gemm-{tile}")
    replace_once(dest / "task3_gemm.cu", "constexpr int TILE = 16;",
                 f"constexpr int TILE = {tile};")
dest = variant("gemm-unpadded")
replace_once(dest / "task3_gemm.cu", "ws[TILE][TILE + 1]", "ws[TILE][TILE]")

for warps in (1, 2, 8, 16):
    dest = variant(f"attn-{warps}")
    replace_once(dest / "task4_attention.cu", "constexpr int ATTN_WARPS = 4;",
                 f"constexpr int ATTN_WARPS = {warps};")
dest = variant("attn-unpadded")
replace_once(dest / "task4_attention.cu", "const int k_pitch = head_dim + 1;",
             "const int k_pitch = head_dim;")
replace_once(dest / "task4_attention.cu", "KV_TILE * (head_dim + 1)",
             "KV_TILE * head_dim")
PY
for variant in gemm-8 gemm-32 gemm-unpadded attn-1 attn-2 attn-8 attn-16 attn-unpadded; do
    test_name=test_task3_gemm
    if [[ "$variant" == attn-* ]]; then test_name=test_task4_attention; fi
    variant_build="$run_dir/$variant"
    run_logged "build-$variant" make -j "${SLURM_CPUS_PER_TASK:-2}" \
        SRC="$run_dir/src-$variant" BUILD="$variant_build" "$variant_build/$test_name"
done
for round in 1 2 3; do
    variants='gemm-8 gemm-16 gemm-32 gemm-unpadded attn-1 attn-2 attn-4 attn-8 attn-16 attn-unpadded'
    if [[ "$round" == 2 ]]; then
        variants='attn-unpadded attn-16 attn-8 attn-4 attn-2 attn-1 gemm-unpadded gemm-32 gemm-16 gemm-8'
    fi
    for variant in $variants; do
        test_name=test_task3_gemm
        if [[ "$variant" == attn-* ]]; then test_name=test_task4_attention; fi
        binary="$run_dir/$variant/$test_name"
        if [[ "$variant" == gemm-16 || "$variant" == attn-4 ]]; then binary="$BUILD/$test_name"; fi
        run_logged "variant-$variant-run-$round" "$binary"
    done
done
sha256sum -c "$run_dir/source.sha256" > "$run_dir/source-integrity.log"
echo 'Task 3/4 validation complete. Default source unchanged by experiments.'
printf 'Logs: %s\n' "$run_dir"
