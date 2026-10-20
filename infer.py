"""Real-time ISL recognition from a webcam, a video file, or images.

    python infer.py                                  # alphabet model, default webcam
    python infer.py --weights phrases --source 1    # phrases model, second camera
    python infer.py --source clip.mp4 --save out.mp4 --no-show
    python infer.py --source path/to/images/        # annotated copies in runs/predict/

Live mode turns detections into text: a sign held steady for --hold frames is
appended to the transcript at the bottom of the window. Lower the hand (or
switch signs) to type the same sign twice.

Keys: q / Esc quit · Space insert space · Backspace delete · Enter clear
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

from common import ROOT, pick_device, resolve_weights


class SignStabilizer:
    """Commit a label once it has been the top detection for `hold` consecutive frames."""

    def __init__(self, hold):
        self.hold = hold
        self.current = None
        self.count = 0
        self.last_committed = None

    def update(self, label):
        if label == self.current:
            self.count += 1
        else:
            self.current, self.count = label, 1
        if label is None:
            self.last_committed = None  # hand dropped: allow the same sign again
            return None
        if self.count == self.hold and label != self.last_committed:
            self.last_committed = label
            return label
        return None

    @property
    def progress(self):
        return 0.0 if self.current is None else min(self.count / self.hold, 1.0)


def draw_hud(img, fps, label, progress, text):
    h, w = img.shape[:2]
    # Top-left: FPS and the sign currently being held, with a hold-progress bar.
    cv2.rectangle(img, (0, 0), (260, 64), (0, 0, 0), -1)
    cv2.putText(img, f"{fps:5.1f} FPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"sign: {label or '-'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (150, 40), (250, 52), (80, 80, 80), -1)
    cv2.rectangle(img, (150, 40), (150 + int(100 * progress), 52), (0, 200, 0), -1)
    # Bottom: running transcript (tail only, so it never overflows).
    cv2.rectangle(img, (0, h - 48), (w, h), (0, 0, 0), -1)
    max_chars = max(10, w // 16)
    tail = text[-max_chars:] if len(text) > max_chars else text
    cv2.putText(img, tail + "_", (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def run_stream(model, source, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {source!r}")

    writer = None
    if args.save:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    # Single-character class names (the alphabet model) are spelled out;
    # word classes are separated by spaces.
    spelling = all(len(n) == 1 for n in model.names.values())
    stabilizer = SignStabilizer(args.hold)
    tokens = []
    fps, last = 0.0, time.perf_counter()
    window = f"ISL - YOLOv11 ({Path(args.weights).stem})"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = result.boxes
        label = model.names[int(boxes.cls[boxes.conf.argmax()])] if len(boxes) else None

        committed = stabilizer.update(label)
        if committed:
            tokens.append(committed if spelling else (" " + committed if tokens else committed))
            if args.no_show:
                print(committed, flush=True)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now

        annotated = draw_hud(result.plot(), fps, label, stabilizer.progress, "".join(tokens))
        if writer:
            writer.write(annotated)
        if not args.no_show:
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                tokens.append(" ")
            elif key in (8, 127) and tokens:
                tokens.pop()
            elif key == 13:
                tokens.clear()

    cap.release()
    if writer:
        writer.release()
        print(f"Saved {args.save}")
    cv2.destroyAllWindows()
    print("Transcript:", "".join(tokens).strip())


def run_images(model, source, args):
    results = model.predict(
        source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(ROOT / "runs" / "predict"),
        name=Path(args.weights).stem,
        stream=True,
        verbose=False,
    )
    for r in results:
        dets = sorted(zip(r.boxes.conf.tolist(), r.boxes.cls.tolist()), reverse=True)
        found = ", ".join(f"{model.names[int(c)]} {p:.2f}" for p, c in dets) or "no sign detected"
        print(f"{Path(r.path).name}: {found}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="alphabet", help="task name (alphabet | phrases) or path to weights")
    parser.add_argument("--source", default="0", help="camera index, video file, image, or folder of images")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hold", type=int, default=8, help="frames a sign must be held before it is typed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mirror", action="store_true", help="flip frames horizontally (selfie view)")
    parser.add_argument("--save", default=None, help="write the annotated stream to this .mp4")
    parser.add_argument("--no-show", action="store_true", help="run headless (no preview window)")
    args = parser.parse_args()

    args.weights = resolve_weights(args.weights)
    args.device = pick_device(args.device)
    model = YOLO(args.weights)

    src = args.source
    suffix = Path(src).suffix[1:].lower()
    if src.isdigit():
        run_stream(model, int(src), args)
    elif suffix in VID_FORMATS:
        run_stream(model, src, args)
    elif Path(src).is_dir() or suffix in IMG_FORMATS:
        run_images(model, src, args)
    else:
        raise SystemExit(f"Unrecognised source {src!r}")


if __name__ == "__main__":
    main()
