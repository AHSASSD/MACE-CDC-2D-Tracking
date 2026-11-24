#!/bin/bash
#SBATCH --job-name=ai6-triplet-path-junction
#SBATCH --partition=gpu_v100
#SBATCH --account=zhangyongchao
#SBATCH --qos=ogsp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=03:00:00
#SBATCH --output=/seu_nvme/home/zhangyongchao/213230965/ai6_t_path_20261122/slurm-%j.log
set -euo pipefail
python3 /seu_nvme/home/zhangyongchao/213230965/ai6_t_path_20261122/runtime/supervisor.py /seu_nvme/home/zhangyongchao/213230965/ai6_t_path_20261122
