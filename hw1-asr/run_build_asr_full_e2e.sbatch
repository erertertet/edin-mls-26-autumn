#!/usr/bin/env bash
#SBATCH --job-name=mls-asr-sm86-build
#SBATCH --partition=Teaching
#SBATCH --nodelist=landonia11
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mls-asr"
export PATH="/opt/cuda-12.8.0/bin:$PATH" NVCC_CCBIN=/usr/bin/g++
run="$PWD/rag/results/asr-full-e2e-a6000-20261117"
mkdir -p "$run/build/sm_86/tmp"
export TMPDIR="$run/build/sm_86/tmp"
make -C hw1-asr BUILD="$run/build/sm_86" SM=86 NVCC=/opt/cuda-12.8.0/bin/nvcc test | tee "$run/kernel-tests.log"
"$HOME/.conda/envs/mls-asr/bin/python" hw1-asr/build_asr_full_e2e_library.py
