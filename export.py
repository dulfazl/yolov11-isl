"""Export trained weights for deployment.

    python export.py --weights alphabet --format onnx      # ONNX Runtime, OpenCV DNN
    python export.py --weights alphabet --format coreml    # iOS / macOS
    python export.py --weights phrases --format tflite     # Android / edge

The exported file is written next to the source weights.
"""

import argparse

from ultralytics import YOLO

from common import resolve_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="alphabet", help="task name (alphabet | phrases) or path to weights")
    parser.add_argument("--format", default="onnx", help="onnx | coreml | tflite | openvino | engine | ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 where the format supports it")
    parser.add_argument("--dynamic", action="store_true", help="dynamic input shapes (ONNX / TensorRT)")
    args = parser.parse_args()

    path = YOLO(resolve_weights(args.weights)).export(
        format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic
    )
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
