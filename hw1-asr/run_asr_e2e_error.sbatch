#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-e2e-error
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=asr-e2e-error-%j.out
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
scratch=$(mktemp -d "/disk/scratch/mls-asr-e2e-$USER-$SLURM_JOB_ID-XXXXXX")
cleanup() {
    status=$?
    case "$scratch" in /disk/scratch/mls-asr-e2e-"$USER"-"$SLURM_JOB_ID"-*) rm -rf -- "$scratch";; esac
    exit "$status"
}
trap cleanup EXIT
export PATH="/opt/cuda-12.8.0/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset OPENROUTER_API_KEY OPENAI_API_KEY ELM_API_KEY
model_cache="$HOME/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512"
mkdir -p "$scratch/hf/models--zai-org--GLM-ASR-Nano-2512/snapshots"
cp -a "$model_cache/refs" "$scratch/hf/models--zai-org--GLM-ASR-Nano-2512/"
cp -aL "$model_cache/snapshots/61ba4e0b3309b6656edea3e93e419f7bd5c61957" "$scratch/hf/models--zai-org--GLM-ASR-Nano-2512/snapshots/"
export HF_HUB_CACHE="$scratch/hf"
export BUILD="$scratch/build" ASR_LIB="$scratch/build/libasr_kernels.so" SM=90
make -C hw1-asr BUILD="$BUILD" SM="$SM" lib
"$HOME/.conda/envs/mls-asr/bin/python" hw1-asr/measure_asr_e2e_error.py \
    --root "$PWD" --out "$PWD/rag/results/asr-e2e-error-20261117"
