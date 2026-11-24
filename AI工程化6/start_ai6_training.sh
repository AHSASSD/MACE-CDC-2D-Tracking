#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
if test -f remote_batch_selection.json; then
  MACE_BATCH_SIZE=$(/usr/bin/python -c 'import json; print(json.load(open("remote_batch_selection.json"))["batch_size"])')
  MACE_VAL_BATCH_SIZE=$(/usr/bin/python -c 'import json; print(json.load(open("remote_batch_selection.json"))["validation_batch_size"])')
  export MACE_BATCH_SIZE MACE_VAL_BATCH_SIZE
fi
command -v flock >/dev/null
if test -f server_runtime_selection.json; then
  exec flock -n joint_training/.training.lock /usr/bin/python -u launch_server_training.py
fi
exec flock -n joint_training/.training.lock /usr/bin/python -u 运行完整流程.py
