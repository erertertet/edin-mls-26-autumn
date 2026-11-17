#!/usr/bin/env bash
# Submit from the PR checkout root. CPU-only retrieval; no ASR or API requests.
#SBATCH --job-name=mls-asr4-normalize
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=asr4-normalize-%j.out
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
scratch=$(mktemp -d "/disk/scratch/mls-asr4-normalize-$USER-$SLURM_JOB_ID-XXXXXX")
trap 'case "$scratch" in /disk/scratch/mls-asr4-normalize-"$USER"-"$SLURM_JOB_ID"-*) rm -rf -- "$scratch";; esac' EXIT
export TMPDIR="$scratch/tmp"
mkdir -p "$TMPDIR" "$scratch/raw"
unset OPENROUTER_API_KEY OPENAI_API_KEY
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python="${RAG_PYTHON:-$HOME/.venvs/mls-rag-pilot/bin/python}"
raw="${RAW_PILOT_RUN:-$PWD/rag/results/asr-four-model-pilot-20261116}"
out="${NORMALIZED_PILOT_OUT:-$PWD/rag/results/asr-four-model-normalized-20261116}"
mkdir -p "$out"
{
  date -u --iso-8601=seconds
  hostname -f
  git rev-parse HEAD
  "$python" -m pip list --format=freeze
} > "$out/slurm-environment-$SLURM_JOB_ID.log" 2>&1
"$python" -m unittest discover -s rag/tests -p test_transcript_normalization.py -v
printf 'Staging the fixed corpus, encoder, tokenizer and saved transcripts.\n'
cp -aL rag/data/rag_corpus "$scratch/corpus"
cp -aL rag/models/enc_a "$scratch/enc_a"
tokenizer=$("$python" -c 'import json,pathlib; print(json.loads((pathlib.Path.home()/".cache/mls-asr-pilot/tokenizer-metadata.json").read_text())["path"])')
cp "$tokenizer" "$scratch/tokenizer.json"
cp -a "$raw/transcripts" "$scratch/raw/"
cp "$raw/config.json" "$raw/selection.json" "$raw/retrieval.jsonl" "$raw/summary.json" "$scratch/raw/"
"$python" rag/extensions/asr_normalization_rerun.py --raw-run "$scratch/raw" --out "$out" \
  --data-dir "$scratch/corpus" --encoder-dir "$scratch/enc_a" --tokenizer-file "$scratch/tokenizer.json"
