#!/bin/bash
#SBATCH --job-name=ai6-full-data-OC-stage1
#SBATCH --partition=gpu_v100
#SBATCH --account=zhangyongchao
#SBATCH --qos=ogsp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --output=/seu_nvme/home/zhangyongchao/213230965/ai6_membership_20261123/full_data_stage1/slurm-%j.log
set -euo pipefail
RUN=/seu_nvme/home/zhangyongchao/213230965/ai6_membership_20261123/full_data_stage1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XDG_RUNTIME_DIR=/tmp PYTHONUNBUFFERED=1
export SLURM_JOB_END_TIME=$(( $(date +%s) + 21600 ))
exec srun --ntasks=1 --container-image="$HOME/pytorch-24.08-ssh.sqsh" --container-mounts=/seu_share:/seu_share,/seu_share2:/seu_share2,/seu_nvme:/seu_nvme --container-workdir="$RUN" --container-writable --no-container-remap-root python -u "$RUN/supervise_full_stage.py"
