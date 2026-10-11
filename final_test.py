from pathlib import Path
import hashlib
import json
import shutil

import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/expanded_checked_960_100_keep_mosaic"
WEIGHTS = RUN / "weights/best.pt"
DATA = ROOT / "toolkit/data/rm_dataset_expanded_checked/data.yaml"
OUTPUT = ROOT / "submission_model"


def main():
    for path in (WEIGHTS, DATA):
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUTPUT.exists():
        raise FileExistsError(f"目录已经存在，避免覆盖：{OUTPUT}")

    # 只评估固定好的最终模型，不更新权重。
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=960,
        batch=8,
        device=0,
        workers=4,
        augment=False,
        plots=True,
        project=str(ROOT / "runs"),
        name="final_test_checked",
        exist_ok=False,
    )

    # 保存最终模型和可追溯的测试记录。
    OUTPUT.mkdir()
    shutil.copy2(WEIGHTS, OUTPUT / "best.pt")

    record = {
        "source_weights": str(WEIGHTS),
        "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "data": str(DATA),
        "split": "test",
        "imgsz": 960,
        "batch": 8,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "evaluation_directory": str(metrics.save_dir),
    }

    (OUTPUT / "test_metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保留最终训练参数与每轮成绩。
    for filename in ("args.yaml", "results.csv"):
        source = RUN / filename
        if source.is_file():
            shutil.copy2(source, OUTPUT / filename)

    print("\n最终测试结果：")
    for key in ("precision", "recall", "mAP50", "mAP50_95"):
        print(f"{key}：{record[key]:.4f}")

    print("\n模型整理目录：", OUTPUT)
    print("测试图表目录：", metrics.save_dir)


if __name__ == "__main__":
    main()