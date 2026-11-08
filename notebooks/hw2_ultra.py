"""Shared code for the Ultralytics models: YOLO (04), RT-DETR (05) and the final pipeline (06).

Sources:
- Ultralytics train/predict API: https://docs.ultralytics.com/modes/train/ , https://docs.ultralytics.com/modes/predict/
- Image slicing idea: SAHI, arXiv:2202.06934 (our simplified version: fixed grid + full frame, merged by NMS).
- Temporal rescoring idea: Seq-NMS, arXiv:1602.08465 (our simplified version: average with matched neighbours).
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms, box_iou

import hw2_eval as E

SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
SEED = 42
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
VAL_STEP = 5
READERS = ThreadPoolExecutor(16)
PREFETCH = ThreadPoolExecutor(1)


def paths():
    if Path("/kaggle/input").exists():
        inp = Path("/kaggle/input")
        data = next(p for p in (inp / "competitions" / SLUG, inp / SLUG) if p.exists())
        prepared = next(inp.glob("**/hw2_prepared/train_coco.json")).parent
        return data, prepared, Path("/kaggle/working")
    root = Path.cwd().parent.parent
    return root / "hw2_data", root / "hw2_prepared", root / "hw2_runs"


def val_frames(data):
    images = json.loads((data / "val" / "annotations_coco.json").read_text())["images"]
    df = pd.DataFrame(images)
    df["path"] = [str(data / "val" / "images" / Path(f).parent.name / Path(f).name) for f in df["file_name"]]
    df["pos"] = df.groupby("video")["frame_id"].rank(method="first").astype(int) - 1
    return df


def test_frames(data):
    df = pd.read_csv(data / "test_images.csv").rename(columns={"image_id": "id", "clip": "video", "frame": "frame_id"})
    df["path"] = [str(data / f) for f in df["file_name"]]
    return df


def write_data_yaml(prepared, val_paths, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_sub.txt").write_text("\n".join(val_paths) + "\n")
    names = [c["name"] for c in json.loads((prepared / "train_coco.json").read_text())["categories"]]
    path = out / "data.yaml"
    path.write_text(f"path: {prepared}\ntrain: images/train\nval: {out / 'val_sub.txt'}\nnames: {json.dumps(names)}\n")
    return path


def best_run(model_dir, names, metric="mAP"):
    scores = {n: json.loads((model_dir / n / "metrics.json").read_text())["val"][metric]
              for n in names if (model_dir / n / "metrics.json").exists()}
    return max(scores, key=scores.get) if scores else names[0]


def to_coco(image_id, xyxy, cls, score):
    image_id = image_id if isinstance(image_id, str) else int(image_id)
    xywh = xyxy.clone()
    xywh[:, 2:] -= xywh[:, :2]
    return [{"image_id": image_id, "category_id": int(c) + 1, "score": round(float(s), 4),
             "bbox": [round(float(v), 2) for v in b]}
            for b, c, s in zip(xywh.tolist(), cls.tolist(), score.tolist())]


def read_images(paths):
    return list(READERS.map(cv2.imread, paths))


def prefetched(groups):
    pending = PREFETCH.submit(read_images, groups[0]) if groups else None
    for k in range(len(groups)):
        images = pending.result()
        if k + 1 < len(groups):
            pending = PREFETCH.submit(read_images, groups[k + 1])
        yield images


def predict(model, frames, imgsz, batch=8, augment=False):
    detections = []
    chunks = [frames.iloc[s:s + batch] for s in range(0, len(frames), batch)]
    for chunk, images in zip(chunks, prefetched([list(c["path"]) for c in chunks])):
        results = model.predict(images, imgsz=imgsz, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                                augment=augment, batch=batch, verbose=False)
        for image_id, r in zip(chunk["id"], results):
            detections += to_coco(image_id, r.boxes.xyxy.cpu(), r.boxes.cls.cpu(), r.boxes.conf.cpu())
    return detections


def tiles(w, h, grid, overlap):
    tw, th = (math.ceil(s / (grid - (grid - 1) * overlap)) for s in (w, h))
    xs = np.linspace(0, w - tw, grid).round().astype(int)
    ys = np.linspace(0, h - th, grid).round().astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def predict_tiled(model, frames, imgsz, grid=2, overlap=0.2, merge_iou=0.6):
    detections = []
    for image_id, (image,) in zip(frames["id"], prefetched([[p] for p in frames["path"]])):
        h, w = image.shape[:2]
        windows = [(0, 0, w, h)] + tiles(w, h, grid, overlap)
        results = model.predict([image[y0:y1, x0:x1] for x0, y0, x1, y1 in windows], imgsz=imgsz, conf=CONF,
                                iou=NMS_IOU, max_det=MAX_DET, batch=len(windows), verbose=False)
        xyxy = torch.cat([r.boxes.xyxy.cpu() + torch.tensor([x0, y0, x0, y0]) for (x0, y0, _, _), r in zip(windows, results)])
        cls = torch.cat([r.boxes.cls.cpu() for r in results])
        conf = torch.cat([r.boxes.conf.cpu() for r in results])
        keep = batched_nms(xyxy, conf, cls.long(), merge_iou)[:MAX_DET]
        detections += to_coco(image_id, xyxy[keep], cls[keep], conf[keep])
    return detections


def temporal_rescore(detections, frames, iou_thr=0.5):
    df = pd.DataFrame(detections)
    xyxy = torch.tensor(df["bbox"].tolist())
    xyxy[:, 2:] += xyxy[:, :2]
    scores = df["score"].to_numpy()
    classes = df["category_id"].to_numpy()
    groups = df.groupby("image_id").indices
    rescored = scores.copy()
    for _, clip in frames.sort_values(["video", "frame_id"]).groupby("video"):
        ids = list(clip["id"])
        for k, image_id in enumerate(ids):
            i = groups.get(image_id)
            if i is None:
                continue
            total, count = scores[i].copy(), 1
            for j in (k - 1, k + 1):
                if not 0 <= j < len(ids):
                    continue
                count += 1
                n = groups.get(ids[j])
                if n is None:
                    continue
                iou = box_iou(xyxy[i], xyxy[n]).numpy()
                iou[classes[i][:, None] != classes[n][None]] = 0
                total += np.where(iou >= iou_thr, scores[n][None], 0).max(1)
            rescored[i] = total / count
    df["score"] = rescored.round(4)
    return df.to_dict("records")


def ms_per_image(model, paths_, imgsz, n=100):
    times = []
    for p in paths_[:n]:
        r = model.predict(p, imgsz=imgsz, conf=CONF, iou=NMS_IOU, max_det=MAX_DET, verbose=False)[0]
        times.append(sum(r.speed.values()))
    return float(np.median(times[5:] or times))


def resumable(out):
    last = out / "weights" / "last.pt"
    return last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1) >= 0


def history(out):
    df = pd.read_csv(out / "results.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"metrics/mAP50-95(B)": "mAP", "metrics/mAP50(B)": "AP50"})
    df.to_csv(out / "history.csv", index=False)
    return df


def finish(model, out, config, imgsz, val, test, data):
    gt = E.load_gt(data / "val" / "annotations_coco.json")
    val_detections = predict(model, val, imgsz)
    test_detections = predict(model, test, imgsz)
    summary, per_class, clips = E.report(gt, val_detections)
    speed = ms_per_image(model, list(val["path"]), imgsz)
    E.write_submission(test_detections, data / "test_images.csv", out / "submission.csv")
    results = {"config": config, "val": summary.to_dict(), "per_class": per_class.to_dict(),
               "clips": clips.to_dict(), "ms_per_image": speed}
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    (out / "val_predictions.json").write_text(json.dumps(val_detections))
    (out / "test_predictions.json").write_text(json.dumps(test_detections))
    return summary, per_class, clips, speed


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def timer():
    start = time.time()
    return lambda: (time.time() - start) / 60
