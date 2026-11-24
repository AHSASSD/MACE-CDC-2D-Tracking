#!/bin/bash
#SBATCH --job-name=ai6-membership-CD-pilot
#SBATCH --partition=gpu_v100
#SBATCH --account=zhangyongchao
#SBATCH --qos=ogsp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
#SBATCH --output=/seu_nvme/home/zhangyongchao/213230965/ai6_membership_20261123/slurm-%j.log
set -euo pipefail
RUN=/seu_nvme/home/zhangyongchao/213230965/ai6_membership_20261123
export XDG_RUNTIME_DIR=/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export SLURM_JOB_END_TIME=$(( $(date +%s) + 21600 ))
nvidia-smi
for attempt in 1 2 3; do
  if srun --ntasks=1 --container-image="$HOME/pytorch-24.08-ssh.sqsh" --container-mounts=/seu_share:/seu_share,/seu_share2:/seu_share2,/seu_nvme:/seu_nvme --container-workdir="$RUN" --container-writable --no-container-remap-root python -u "$RUN/train_pilot.py" --data /seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_event_dataset --cache /seu_nvme/home/zhangyongchao/213230965/ai6_graph_cache_v1 --out "$RUN/pilot"; then exit 0; fi
  echo "pilot failed; recovery attempt $attempt/3"
done
exit 1
