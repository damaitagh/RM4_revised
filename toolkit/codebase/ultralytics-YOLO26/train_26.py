"""Train YOLO26 on the RM2025 car dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("your_yaml_path")


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset YAML path")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolo26n.pt", help="pretrained weights")
    parser.add_argument(
        "--cfg",
        type=Path,
        default=PROJECT_ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
        help="model YAML used with --from-scratch",
    )
    parser.add_argument("--from-scratch", action="store_true", help="initialize the model from --cfg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, or cpu")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of training images to use")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs/detect")
    parser.add_argument("--name", default="rm2025_part1")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    model_source = opt.cfg if opt.from_scratch else opt.weights
    model = YOLO(str(model_source))
    model.info()
    model.train(
        data=str(opt.data),
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        workers=opt.workers,
        batch=opt.batch,
        device=opt.device,
        fraction=opt.fraction,
        val=False,
        project=str(opt.project),
        name=opt.name,
        exist_ok=opt.exist_ok,
    )


if __name__ == "__main__":
    main(parse_opt())
