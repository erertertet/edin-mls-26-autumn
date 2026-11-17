#!/usr/bin/env bash
# Submit from the frozen PR checkout; all 475 queries, four ASR arms.
#SBATCH --job-name=mls-asr4-full
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=asr4-full-%j.out
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
scratch=$(mktemp -d "/disk/scratch/mls-asr4-full-$USER-$SLURM_JOB_ID-XXXXXX")
cleanup() {
  status=$?
  case "$scratch" in /disk/scratch/mls-asr4-full-"$USER"-"$SLURM_JOB_ID"-*) rm -rf -- "$scratch";; esac
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$scratch/tmp"
mkdir -p "$TMPDIR"
export PATH="/opt/cuda-12.8.0/bin:$PATH"
export RAG_ASR_PYTHON="$HOME/.conda/envs/mls-asr/bin/python"
export ASR_CUDA=all HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python="$HOME/.venvs/mls-rag-pilot/bin/python"
out="${FULL_ASR_OUT:-$PWD/rag/results/asr-four-model-full-20261117}"
mkdir -p "$out"
{
  date -u --iso-8601=seconds
  hostname -f
  git rev-parse HEAD
  nvidia-smi -L
  nvcc --version
  "$python" -m pip list --format=freeze
} > "$out/slurm-environment-$SLURM_JOB_ID.log" 2>&1
"$python" -m unittest discover -s rag/tests -p test_asr_retrieval_pilot.py -v
"$python" -m unittest discover -s rag/tests -p test_transcript_normalization.py -v
"$python" -m unittest discover -s rag/tests -p test_asr_full_summary.py -v
printf 'Staging the fixed course inputs to node-local scratch.\n'
cp -aL rag/data/rag_corpus "$scratch/corpus"
cp -aL rag/models/enc_a "$scratch/enc_a"
tokenizer=$("$python" -c 'import json,pathlib;print(json.loads((pathlib.Path.home()/".cache/mls-asr-pilot/tokenizer-metadata.json").read_text())["path"])')
cp "$tokenizer" "$scratch/tokenizer.json"
model_cache="$HOME/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512"
mkdir -p "$scratch/hf/models--zai-org--GLM-ASR-Nano-2512"
cp -a "$model_cache/refs" "$scratch/hf/models--zai-org--GLM-ASR-Nano-2512/"
cp -aL "$model_cache/snapshots" "$scratch/hf/models--zai-org--GLM-ASR-Nano-2512/"
export HF_HUB_CACHE="$scratch/hf"
export BUILD="$scratch/build" ASR_LIB="$scratch/build/libasr_kernels.so"
make -C hw1-asr BUILD="$BUILD" lib
sha256sum "$ASR_LIB" > "$out/library-$SLURM_JOB_ID.sha256"
"$python" rag/extensions/asr_full_experiment.py --out "$out" --data-dir "$scratch/corpus" \
  --encoder-dir "$scratch/enc_a" --tokenizer-file "$scratch/tokenizer.json" "$@"
