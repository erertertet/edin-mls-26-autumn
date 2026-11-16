#!/usr/bin/env bash
# Submit from the repository root: sbatch rag/extensions/run_asr_retrieval_pilot.sbatch
#SBATCH --job-name=mls-asr4-pilot
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=asr4-pilot-%j.out

set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}"
case "$(hostname -f)" in
  *.inf.ed.ac.uk) ;;
  *) printf 'Run only in an Edinburgh Slurm allocation.\n' >&2; exit 1 ;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f rag/extensions/asr_retrieval_pilot.py
export PATH="/opt/cuda-12.8.0/bin:$PATH"
export RAG_ASR_PYTHON="${RAG_ASR_PYTHON:-$HOME/.conda/envs/mls-asr/bin/python}"
export ASR_CUDA=all
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
rag_python="${RAG_PYTHON:-$HOME/.venvs/mls-rag-pilot/bin/python}"
out="${PILOT_OUT:-$PWD/rag/results/asr-four-model-pilot-20261116}"
mkdir -p "$out"
{
  date -u --iso-8601=seconds
  hostname -f
  printf 'Slurm job: %s\n' "$SLURM_JOB_ID"
  git rev-parse HEAD
  nvidia-smi -L
  nvcc --version
  "$rag_python" -m pip list --format=freeze
} > "$out/slurm-environment-$SLURM_JOB_ID.log" 2>&1
"$rag_python" -m unittest discover -s rag/tests -p test_asr_retrieval_pilot.py -v
export BUILD="$PWD/hw1-asr/build/asr-pilot-$SLURM_JOB_ID"
export ASR_LIB="$BUILD/libasr_kernels.so"
make -C hw1-asr BUILD="$BUILD" lib
sha256sum "$ASR_LIB" > "$out/library-$SLURM_JOB_ID.sha256"
"$rag_python" rag/extensions/asr_retrieval_pilot.py --out "$out" --validate-only
"$rag_python" rag/extensions/asr_retrieval_pilot.py --out "$out" --budget-usd 1
