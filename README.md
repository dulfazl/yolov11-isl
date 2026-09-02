# Indian Sign Language Recognition

Real-time recognition of Indian Sign Language (ISL) signs with a fine-tuned YOLOv11 detection model.

> **Status — work in progress.** Dataset preparation, training code and weights are being pushed here. The notes below describe the approach; results will be added once training runs are complete.

## Why ISL

Most published sign-language recognition work targets ASL. Indian Sign Language has a different alphabet, is largely two-handed, and has far less annotated public data — which makes it both a harder problem and a more useful one to work on.

## Approach

- **Model** — YOLOv11, fine-tuned for frame-level detection of hand signs
- **Target** — live inference on webcam input rather than offline classification of pre-cropped images
- **Pipeline** — dataset preparation and augmentation → training → evaluation (mAP, per-class accuracy) → real-time inference loop

## Roadmap

- [ ] Dataset collection and annotation
- [ ] Baseline training run and evaluation
- [ ] Real-time webcam inference
- [ ] Published weights and reproducible training script

---

Built by [Muhammed Dulfazal](https://github.com/dulfazl) — MSc AI & ML, VIT Vellore.
