#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
OUT=../../hw2_runs/smoke
mkdir -p "$OUT"
status=0

smoke() {
  local nb=$1 name=$2
  shift 2
  if $PY -m papermill "$nb" "$OUT/$name.ipynb" -k hw2 --cwd . -p SMOKE_TEST True "$@" > "$OUT/$name.log" 2>&1; then
    echo "ok     $name"
  else
    echo "FAILED $name (see $OUT/$name.log)"
    status=1
  fi
}

smoke 04_yolo.ipynb yolo-baseline -p EXPERIMENT baseline
smoke 04_yolo.ipynb yolo-exp1 -p EXPERIMENT exp1_imgsz1280
smoke 04_yolo.ipynb yolo-exp2 -p EXPERIMENT exp2_nwd
smoke 06_pipeline.ipynb pipeline
smoke 03_retinanet_train.ipynb retinanet-exp2 -p EXPERIMENT exp2_giou
smoke 05_rtdetr.ipynb rtdetr-exp1 -p EXPERIMENT exp1_queries500
smoke 05_rtdetr.ipynb rtdetr-exp2 -p EXPERIMENT exp2_giou
exit $status
