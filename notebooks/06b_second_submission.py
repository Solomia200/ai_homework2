"""Second Kaggle candidate: YOLO11m from 06_pipeline (part B) with tiled inference.

06_pipeline picks one final variant by mAP_test_like (YOLO11s + TTA). YOLO11m + tiles is best by full-val mAP,
so it is kept as the second final submission. Run from notebooks/: python 06b_second_submission.py
"""
import json

import pandas as pd
from ultralytics import YOLO

import hw2_eval as E
import hw2_ultra as U

DATA, _, RUNS = U.paths()
pipeline = RUNS / "pipeline"
out = pipeline / "final_yolo11m_tiles"
out.mkdir(exist_ok=True)
imgsz = json.loads((pipeline / "final" / "metrics.json").read_text())["imgsz"]

model = YOLO(pipeline / "yolo11m" / "weights" / "best.pt")
test_detections = U.predict_tiled(model, U.test_frames(DATA), imgsz)
E.write_submission(test_detections, DATA / "test_images.csv", out / "submission.csv")

val = pd.read_csv(pipeline / "strategies_B.csv", index_col=0).loc["tiles"].to_dict()
(out / "metrics.json").write_text(json.dumps({"model": "yolo11m", "variant": "tiles", "imgsz": imgsz, "val": val}, indent=1))
print(out / "submission.csv", val["mAP"], val["mAP_test_like"])
