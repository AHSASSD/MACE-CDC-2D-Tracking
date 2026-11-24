#!/bin/bash
set -euo pipefail
RUN=/seu_nvme/home/zhangyongchao/213230965/ai6_et_20261122
DATA=/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_event_dataset
WEIGHT=/seu_share/home/zhangyongchao/213230965/network_datasets/ai6/joint_training/evaluation_snapshot_epoch126_1794052868/best_interaction_network_joint_no_time.pth
export XDG_RUNTIME_DIR=/tmp ENROOT_DATA_PATH="$HOME/.local/share/enroot"
export MACE_GRAPH_CACHE=/seu_nvme/home/zhangyongchao/213230965/ai6_graph_cache_v1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
MODE=${1:?stage required}
if [[ "$MODE" == prepare_* ]]; then
    PROGRAM=(python -u "$RUN/runtime/STEP1_准备特征.py" --data "$DATA" --weight "$WEIGHT" --cache "$RUN/cache" --splits "$RUN/runtime/split_manifest.json" --mode "${MODE#prepare_}")
elif [[ "$MODE" == pilot ]]; then
    PROGRAM=(python -u "$RUN/runtime/STEP2_训练双路线.py" --data "$DATA" --cache "$RUN/cache" --splits "$RUN/runtime/split_manifest.json" --out "$RUN/pilot" --mode pilot --epochs 10 --member-count 16)
else
    PROGRAM=(python -u "$RUN/runtime/STEP2_训练双路线.py" --data "$DATA" --cache "$RUN/cache" --splits "$RUN/runtime/split_manifest.json" --out "$RUN/full" --mode full --epochs 60 --member-count 200)
fi
exec srun --ntasks=1 --container-name=ai6_training_1530118 \
    --container-image="$HOME/pytorch-24.08-ssh.sqsh" \
    --container-mounts=/seu_share:/seu_share,/seu_share2:/seu_share2,/seu_nvme:/seu_nvme \
    --container-workdir="$RUN/runtime" --container-writable --no-container-remap-root "${PROGRAM[@]}"
