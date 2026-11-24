#!/bin/bash
set -euo pipefail
RUN=/seu_nvme/home/zhangyongchao/213230965/ai6_t_path_20261122
OLD=/seu_nvme/home/zhangyongchao/213230965/ai6_et_20261122
export AI6_T_BASE="$OLD/runtime"
export AI6_DATA=/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_event_dataset
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XDG_RUNTIME_DIR=/tmp ENROOT_DATA_PATH="$HOME/.local/share/enroot"
MODE=${1:?stage}
if [[ "$MODE" == train ]]; then
 PROGRAM=(python -u "$RUN/runtime/train.py" --cache "$OLD/cache" --out "$RUN/training" --splits "$OLD/runtime/split_manifest.json" --initial "$OLD/full/T_best_members.pth" --epochs 30)
else
 WEIGHT="$OLD/full/T_best_members.pth"
 if [[ "$MODE" == focused_members ]]; then WEIGHT="$RUN/training/T_best_junction.pth"; fi
 PROGRAM=(python -u "$RUN/runtime/validate_members.py" --cache "$OLD/cache" --weights "$WEIGHT" --splits "$OLD/runtime/split_manifest.json" --out "$RUN/$MODE")
fi
exec srun --ntasks=1 --container-name=ai6_training_1530118 --container-image="$HOME/pytorch-24.08-ssh.sqsh" --container-mounts=/seu_share:/seu_share,/seu_share2:/seu_share2,/seu_nvme:/seu_nvme --container-workdir="$RUN/runtime" --container-writable --no-container-remap-root "${PROGRAM[@]}"
