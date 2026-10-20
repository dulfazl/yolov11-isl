"""Shared paths, model registry and device selection for the ISL scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each task bundles its data config, published weights and Roboflow source.
TASKS = {
    "alphabet": {
        "data": ROOT / "configs" / "alphabet.yaml",
        "weights": ROOT / "weights" / "isl-alphabet-yolo11n.pt",
        "dataset_dir": ROOT / "datasets" / "isl-alphabet",
        "roboflow": ("project-ze4h3", "indian-sign-language-detection-nv2xc", 1),
    },
    "phrases": {
        "data": ROOT / "configs" / "phrases.yaml",
        "weights": ROOT / "weights" / "isl-phrases-yolo11n.pt",
        "dataset_dir": ROOT / "datasets" / "isl-phrases",
        "roboflow": ("rohit-sjsqg", "rohit_isl-115", 4),
    },
}


def pick_device(device=None):
    """Return the requested device, or the fastest one available (CUDA > MPS > CPU)."""
    if device not in (None, "", "auto"):
        return device
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_data(data):
    """Accept a task name ("alphabet") or a path to a data YAML; return an absolute path.

    Ultralytics resolves relative dataset paths against the YAML's directory only
    when the YAML path itself is absolute, so always hand it an absolute path.
    """
    if data in TASKS:
        return str(TASKS[data]["data"])
    return str(Path(data).resolve())


def resolve_weights(weights):
    """Accept a task name ("alphabet") or a path to a .pt / exported model."""
    if weights in TASKS:
        return str(TASKS[weights]["weights"])
    return str(weights)
