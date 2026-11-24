#!/bin/bash
#SBATCH --job-name=ai6-independent-E-T
#SBATCH --partition=gpu_v100
#SBATCH --account=zhangyongchao
#SBATCH --qos=ogsp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --output=/seu_nvme/home/zhangyongchao/213230965/ai6_et_20261122/slurm-%j.log
set -euo pipefail
RUN=/seu_nvme/home/zhangyongchao/213230965/ai6_et_20261122
date -Is
hostname
nvidia-smi
python3 "$RUN/pipeline.py" "$RUN"
