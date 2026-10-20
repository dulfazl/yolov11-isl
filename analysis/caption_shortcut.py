"""Does the alphabet model read the printed caption instead of the hand?

About 60% of the alphabet dataset are frames from teaching videos with the
answer printed in a white band under the signer (bottom ~17% of the image).
This compares top-1 accuracy on those frames when

  * the image is unchanged,
  * the caption band is painted over (hand still fully visible), and
  * everything *except* the caption band is painted over (no hand at all).

If the model leaned on the caption, the second number would drop and the third
would stay well above chance (1/35).

    python analysis/caption_shortcut.py --split test
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset, img2label_paths  # noqa: E402

from common import ROOT, pick_device, resolve_data, resolve_weights  # noqa: E402
from evaluate import ground_truth_class, list_images  # noqa: E402

BAND = 0.17  # caption band height as a fraction of the image


def hide_caption(img):
    img = img.copy()
    img[int(img.shape[0] * (1 - BAND)) :] = 255
    return img


def caption_only(img):
    img = img.copy()
    img[: int(img.shape[0] * (1 - BAND))] = 255
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = YOLO(resolve_weights("alphabet"))
    device = pick_device(args.device)
    images = list_images(check_det_dataset(resolve_data("alphabet"))[args.split])
    gts = [ground_truth_class(lbl) for lbl in img2label_paths([str(p) for p in images])]

    conditions = {"original": lambda im: im, "caption hidden": hide_caption, "caption only": caption_only}
    groups = {"video frames (captioned)": lambda p: "_mp4-" in p.name, "photos (no caption)": lambda p: "_mp4-" not in p.name}
    tally = {g: Counter() for g in groups}
    totals = Counter()

    for path, gt in zip(images, gts):
        if gt is None:
            continue
        group = next(g for g, match in groups.items() if match(path))
        totals[group] += 1
        img = cv2.imread(str(path))
        for cond, fn in conditions.items():
            r = model.predict(fn(img), conf=args.conf, device=device, verbose=False)[0]
            pred = int(r.boxes.cls[r.boxes.conf.argmax()]) if len(r.boxes) else None
            tally[group][cond] += pred == gt

    report = {g: {"images": totals[g], **{c: tally[g][c] / totals[g] for c in conditions}} for g in groups if totals[g]}
    print(f"\nTop-1 accuracy on alphabet/{args.split} (conf >= {args.conf})")
    print(f"{'group':28s} {'images':>6s} " + " ".join(f"{c:>15s}" for c in conditions))
    for g, row in report.items():
        print(f"{g:28s} {row['images']:6d} " + " ".join(f"{row[c]:15.3f}" for c in conditions))

    out = ROOT / "results" / "alphabet" / f"caption_shortcut_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "conf": args.conf, "band": BAND, "groups": report}, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
