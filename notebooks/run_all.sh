#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=../../hw2_runs
mkdir -p "$RUNS/logs" "$RUNS/done"

until [ -f ../../hw2_data/test_images.csv ] && [ -z "$(ls ../../hw2_data/*.zip 2>/dev/null)" ]; do sleep 30; done

run() {
  local nb=$1 out=$2
  shift 2
  if [ -f "$RUNS/done/$out" ]; then
    echo "$(date '+%F %T') skip $out"
    return
  fi
  echo "$(date '+%F %T') start $out"
  if $PY -m papermill "$nb" "$out.ipynb" -k hw2 --cwd . "$@" > "$RUNS/logs/$out.log" 2>&1; then
    touch "$RUNS/done/$out"
    echo "$(date '+%F %T') done $out"
  else
    echo "$(date '+%F %T') FAILED $out (see $RUNS/logs/$out.log)"
  fi
}

if [ ! -f ../../hw2_prepared/train_coco.json ]; then
  echo "$(date '+%F %T') start 01_data"
  $PY -m papermill 01_data.ipynb "$RUNS/01_data_local.ipynb" -k hw2 --cwd . > "$RUNS/logs/01_data.log" 2>&1 \
    || { echo "FAILED 01_data (see $RUNS/logs/01_data.log)"; exit 1; }
fi

run 04_yolo.ipynb hw2-yolo-baseline -p EXPERIMENT baseline
run 04_yolo.ipynb hw2-yolo-exp1 -p EXPERIMENT exp1_imgsz1280
run 04_yolo.ipynb hw2-yolo-exp2 -p EXPERIMENT exp2_nwd
run 06_pipeline.ipynb hw2-pipeline
run 03_retinanet_train.ipynb hw2-retinanet-baseline -p EXPERIMENT baseline
run 03_retinanet_train.ipynb hw2-retinanet-exp1 -p EXPERIMENT exp1_anchors
run 03_retinanet_train.ipynb hw2-retinanet-exp2 -p EXPERIMENT exp2_giou
run 05_rtdetr.ipynb hw2-rtdetr-baseline -p EXPERIMENT baseline
run 05_rtdetr.ipynb hw2-rtdetr-exp1 -p EXPERIMENT exp1_queries500
run 05_rtdetr.ipynb hw2-rtdetr-exp2 -p EXPERIMENT exp2_giou
rm -f "$RUNS/done/hw2-summary"
run 07_summary.ipynb hw2-summary
echo "$(date '+%F %T') all finished"
