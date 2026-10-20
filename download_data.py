"""Download the ISL datasets from Roboflow Universe in YOLOv11 format.

Needs a free Roboflow account: put your API key in ROBOFLOW_API_KEY
(https://app.roboflow.com/settings/api), then run

    python download_data.py alphabet
    python download_data.py phrases
    python download_data.py all

Uses Roboflow's REST export endpoint directly rather than the `roboflow`
package, which pins opencv-python-headless and would break the webcam window.
Alternatively download the "YOLOv11" zip from the dataset page and unzip it
into datasets/isl-alphabet or datasets/isl-phrases.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from common import TASKS

API_URL = "https://api.roboflow.com"


def export_link(workspace, project, version, api_key):
    """Ask Roboflow for the YOLOv11 export of a dataset version, waiting while it is generated."""
    url = f"{API_URL}/{workspace}/{project}/{version}/yolov11?api_key={api_key}&nocache=true"
    while True:
        try:
            with urllib.request.urlopen(url) as resp:
                status, payload = resp.status, json.load(resp)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Roboflow API error {e.code}: {e.read().decode(errors='replace')}")
        if status == 202 or payload.get("ready") is False:
            print(f"\r  export in progress: {payload.get('progress', 0) * 100:.0f}%", end="", flush=True)
            time.sleep(2)
            continue
        return payload["export"]["link"]


def extract(zip_path, dest):
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (dest / member).resolve().is_relative_to(dest):
                raise SystemExit(f"Refusing to extract unsafe path {member!r}")
        zf.extractall(dest)


def download(task, api_key):
    workspace, project, version = TASKS[task]["roboflow"]
    dest = TASKS[task]["dataset_dir"]
    if (dest / "train" / "images").is_dir():
        print(f"[{task}] already present at {dest} — skipping")
        return

    print(f"[{task}] requesting {workspace}/{project} v{version}")
    link = export_link(workspace, project, version, api_key)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "dataset.zip"
        print(f"[{task}] downloading -> {dest}")
        urllib.request.urlretrieve(link, zip_path)
        extract(zip_path, dest)
    n = len(list((dest / "train" / "images").glob("*")))
    # The zip's own data.yaml has Colab-style paths; the scripts use configs/<task>.yaml.
    print(f"[{task}] done: {n} training images. Use --data {task}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=[*TASKS, "all"])
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="defaults to $ROBOFLOW_API_KEY")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("No Roboflow API key. Set ROBOFLOW_API_KEY or pass --api-key.")
    for task in TASKS if args.task == "all" else [args.task]:
        download(task, args.api_key)


if __name__ == "__main__":
    main()
