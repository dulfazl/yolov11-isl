"""Evaluate a trained model on a dataset split.

Reports detection metrics (precision, recall, mAP50, mAP50-95) plus an
image-level top-1 accuracy: for each image, does the most confident detection
carry the right sign? That second number is closer to what a user of the live
demo experiences.

    python evaluate.py --weights alphabet            # test split, data inferred
    python evaluate.py --weights phrases --split val
    python evaluate.py --weights runs/train/alphabet/weights/best.pt --data alphabet

Writes metrics.json, per_class.csv and Ultralytics' plots to results/<task>/<split>/.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

from common import ROOT, TASKS, pick_device, resolve_data, resolve_weights


def list_images(source):
    """Image paths for a split entry (a directory, a .txt list, or a list of either)."""
    sources = source if isinstance(source, list) else [source]
    images = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            images += [f for f in p.rglob("*") if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            images += [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
    return sorted(images)


def ground_truth_class(label_path):
    """Dominant class in a YOLO label file, or None when the image has no labels."""
    if not Path(label_path).exists():
        return None
    ids = [int(line.split()[0]) for line in Path(label_path).read_text().splitlines() if line.strip()]
    return Counter(ids).most_common(1)[0][0] if ids else None


def image_accuracy(model, images, conf, imgsz, device, batch):
    """Per-class (correct, total) counts for top-1 detection vs. ground-truth class."""
    labels = img2label_paths([str(p) for p in images])
    pairs = [(img, ground_truth_class(lbl)) for img, lbl in zip(images, labels)]
    pairs = [(img, gt) for img, gt in pairs if gt is not None]

    counts = {}
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        results = model.predict([str(p) for p, _ in chunk], conf=conf, imgsz=imgsz, device=device, verbose=False)
        for (_, gt), r in zip(chunk, results):
            pred = int(r.boxes.cls[r.boxes.conf.argmax()]) if len(r.boxes) else None
            correct, total = counts.get(gt, (0, 0))
            counts[gt] = (correct + (pred == gt), total + 1)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="alphabet", help="task name (alphabet | phrases) or path to weights")
    parser.add_argument("--data", default=None, help="task name or data YAML (defaults to the --weights task)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for top-1 accuracy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="output folder (default results/<task>/<split>)")
    args = parser.parse_args()

    data_arg = args.data or (args.weights if args.weights in TASKS else None)
    if data_arg is None:
        parser.error("--data is required when --weights is a path")
    task = data_arg if data_arg in TASKS else Path(data_arg).stem
    data = resolve_data(data_arg)
    out = Path(args.out) if args.out else ROOT / "results" / task / args.split
    device = pick_device(args.device)

    model = YOLO(resolve_weights(args.weights))
    names = model.names

    # 1. Detection metrics (mAP etc.) via Ultralytics' validator.
    metrics = model.val(
        data=data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    det_per_class = {}
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        det_per_class[int(c)] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}

    # 2. Image-level top-1 accuracy.
    split_key = "val" if args.split == "val" else args.split
    images = list_images(check_det_dataset(data)[split_key])
    acc_counts = image_accuracy(model, images, args.conf, args.imgsz, device, args.batch)
    n_correct = sum(c for c, _ in acc_counts.values())
    n_images = sum(t for _, t in acc_counts.values())

    per_class = []
    for c in sorted(set(det_per_class) | set(acc_counts)):
        correct, total = acc_counts.get(c, (0, 0))
        row = {"class": names[c], "images": total, "top1_accuracy": correct / total if total else None}
        row.update(det_per_class.get(c, {}))
        per_class.append(row)

    weights_path = Path(resolve_weights(args.weights)).resolve()
    summary = {
        "weights": str(weights_path.relative_to(ROOT) if weights_path.is_relative_to(ROOT) else weights_path),
        "data": task,
        "split": args.split,
        "images": n_images,
        "classes_present": len(per_class),
        "precision": box.mp,
        "recall": box.mr,
        "mAP50": box.map50,
        "mAP50-95": box.map,
        "top1_accuracy": n_correct / n_images if n_images else None,
        "top1_conf_threshold": args.conf,
        "per_class": per_class,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    with open(out / "per_class.csv", "w", newline="") as f:
        fields = ["class", "images", "top1_accuracy", "precision", "recall", "mAP50", "mAP50-95"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_class:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    print(f"\n{task} / {args.split}: {n_images} images, {len(per_class)} classes")
    print(f"  precision {box.mp:.3f}  recall {box.mr:.3f}  mAP50 {box.map50:.3f}  mAP50-95 {box.map:.3f}")
    print(f"  top-1 image accuracy {summary['top1_accuracy']:.3f}  (conf >= {args.conf})")
    weakest = sorted((r for r in per_class if r["images"]), key=lambda r: r["top1_accuracy"])[:5]
    print("  weakest classes: " + ", ".join(f"{r['class']} {r['top1_accuracy']:.2f} ({r['images']} imgs)" for r in weakest))
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()
