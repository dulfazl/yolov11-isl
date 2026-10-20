"""Fine-tune YOLOv11 on an ISL dataset.

Defaults reproduce the published weights (yolo11n, 50 epochs, 640 px, batch 16):

    python train.py --data alphabet
    python train.py --data phrases --epochs 100 --model yolo11s.pt
"""

import argparse
import shutil

from ultralytics import YOLO

from common import ROOT, TASKS, pick_device, resolve_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="alphabet", help="task name (alphabet | phrases) or path to a data YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=100, help="early-stopping patience in epochs")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | 0 | 0,1 ...")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name under runs/train (defaults to the task name)")
    parser.add_argument("--publish", action="store_true", help="copy best.pt over the task's file in weights/")
    args = parser.parse_args()

    name = args.name or (args.data if args.data in TASKS else "custom")
    model = YOLO(args.model)
    model.train(
        data=resolve_data(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=pick_device(args.device),
        workers=args.workers,
        seed=args.seed,
        project=str(ROOT / "runs" / "train"),
        name=name,
        exist_ok=False,
    )

    best = model.trainer.best
    print(f"\nBest weights: {best}")
    if args.publish:
        if args.data not in TASKS:
            raise SystemExit("--publish only works with a task name for --data")
        dest = TASKS[args.data]["weights"]
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(best, dest)
        print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
