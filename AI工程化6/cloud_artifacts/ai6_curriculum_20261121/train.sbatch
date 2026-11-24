#!/bin/bash
#SBATCH --job-name=ai6-curriculum-recall
#SBATCH --partition=gpu_v100
#SBATCH --account=zhangyongchao
#SBATCH --qos=ogsp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --output=/seu_nvme/home/zhangyongchao/213230965/ai6_curriculum_20261121/slurm-%j.log
set -euo pipefail
RUN=/seu_nvme/home/zhangyongchao/213230965/ai6_curriculum_20261121
date -Is
hostname
nvidia-smi
bash "$RUN/run_step.sh" preflight
python3 "$RUN/supervise_remote.py" "$RUN"
