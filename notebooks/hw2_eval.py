"""Shared evaluator and submission writer for all HW2 models.

Detections are lists of dicts in COCO result format:
{"image_id": ..., "category_id": 1..8, "bbox": [x, y, w, h] in original pixels, "score": ...}.

Sources:
- faster-coco-eval (https://github.com/MiXaiLL76/faster_coco_eval) and pycocotools
  (https://github.com/cocodataset/cocoapi) do the COCO matching and accumulation.
- COCO metrics definition: https://cocodataset.org/#detection-eval. Ignored regions are iscrowd=1,
  as the assignment recommends.
- mAP is averaged from eval["precision"] with maxDets=500 (VisDrone protocol, arXiv:2001.06303, and the
  competition's 500-detection limit). pycocotools' summarize() hard-codes maxDets=100 for its headline AP
  (cocoeval.py, _summarizeDets).
- Submission format follows the competition data description.
"""
import contextlib
import io

import numpy as np
import pandas as pd

MAX_DETS = 500
TEST_LIKE_MAX_WIDTH = 1920
AREAS = ["all", "small", "medium", "large"]


def _backend(name):
    if name == "faster":
        from faster_coco_eval import COCO, COCOeval_faster
        return COCO, COCOeval_faster
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    return COCO, COCOeval


def load_gt(path, backend="faster"):
    coco_cls, _ = _backend(backend)
    with contextlib.redirect_stdout(io.StringIO()):
        return coco_cls(str(path))


def load_detections(gt, detections, backend="faster"):
    if detections:
        with contextlib.redirect_stdout(io.StringIO()):
            return gt.loadRes(list(detections))
    coco_cls, _ = _backend(backend)
    empty = coco_cls()
    empty.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
    with contextlib.redirect_stdout(io.StringIO()):
        empty.createIndex()
    return empty


def _precision(gt, dt, image_ids, backend):
    _, eval_cls = _backend(backend)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = eval_cls(gt, dt, "bbox")
        ev.params.maxDets = [1, 100, MAX_DETS]
        if image_ids is not None:
            ev.params.imgIds = [int(i) for i in image_ids]
        ev.evaluate()
        ev.accumulate()
    return ev.eval["precision"], ev.params.catIds


def _ap(precision, iou_index=None, area="all", class_index=None):
    p = precision[:, :, :, AREAS.index(area), -1]
    if iou_index is not None:
        p = p[iou_index:iou_index + 1]
    if class_index is not None:
        p = p[:, :, class_index]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def metrics(gt, dt, image_ids=None, backend="faster"):
    precision, cat_ids = _precision(gt, dt, image_ids, backend)
    summary = {"mAP": _ap(precision), "AP50": _ap(precision, iou_index=0), "AP75": _ap(precision, iou_index=5)}
    summary.update({f"AP_{area}": _ap(precision, area=area) for area in AREAS[1:]})
    per_class = {gt.cats[c]["name"]: _ap(precision, class_index=k) for k, c in enumerate(cat_ids)}
    return summary, per_class


def report(gt, detections, backend="faster"):
    dt = load_detections(gt, detections, backend)
    images = pd.DataFrame(gt.dataset["images"])
    summary, per_class = metrics(gt, dt, backend=backend)
    test_like = images.loc[images["width"] <= TEST_LIKE_MAX_WIDTH, "id"]
    summary["mAP_test_like"] = metrics(gt, dt, test_like, backend)[0]["mAP"]
    clips = {video: metrics(gt, dt, ids, backend)[0]["mAP"] for video, ids in images.groupby("video")["id"]}
    return pd.Series(summary), pd.Series(per_class), pd.Series(clips)


def top_detections(detections, k=MAX_DETS):
    df = pd.DataFrame(detections)
    return df.sort_values("score", ascending=False).groupby("image_id", sort=False).head(k)


def write_submission(detections, test_images_csv, path):
    test = pd.read_csv(test_images_csv)
    df = top_detections(detections)
    x, y, w, h = np.array(df["bbox"].tolist()).round(1).T
    df["group"] = [f"{c} {s:.4f} {a} {b} {c2} {d}" for c, s, a, b, c2, d in
                   zip(df["category_id"] - 1, df["score"], x, y, w, h)]
    strings = df.groupby("image_id")["group"].agg(" ".join)
    submission = pd.DataFrame({"image_id": test["image_id"],
                               "PredictionString": test["image_id"].map(strings).fillna("")})
    submission.to_csv(path, index=False)
    return submission
