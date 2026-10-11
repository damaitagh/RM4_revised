from pathlib import Path
import csv
import json
import shutil

import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "toolkit/data"

WEIGHTS = ROOT / "runs/baseline_yolo26n/weights/best.pt"
MANIFEST = DATA_ROOT / "rm_dataset/split_manifest.json"
OUTPUT = DATA_ROOT / "annotation_work"

# 这是预标注的起始阈值，并非已经调优的最佳值。
CONF = 0.25


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"找不到模型：{WEIGHTS}")

    # 防止覆盖后续人工修改过的标签。
    if OUTPUT.exists():
        raise FileExistsError(f"输出目录已存在，停止操作：{OUTPUT}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    images = [
        DATA_ROOT / name
        for name in manifest["unlabeled_candidates"]
    ]

    assert len(images) == 301, f"图片数量不符：{len(images)}"
    assert len({p.stem for p in images}) == len(images), "文件名冲突"

    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("实际使用的 Ultralytics：", ultralytics.__file__)
    model = YOLO(str(WEIGHTS))
    print("模型类别：", model.names)

    for folder in ["images", "labels", "previews"]:
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

    # 记录此次预标注用的模型和参数。
    config = {
        "weights": str(WEIGHTS),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_source": ultralytics.__file__,
        "conf": CONF,
        "imgsz": 640,
        "image_count": len(images),
    }
    (OUTPUT / "prediction_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_boxes = 0
    no_detection = 0

    with (OUTPUT / "review_status.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image", "predicted_boxes", "min_confidence",
            "status", "notes"
        ])

        for index, path in enumerate(images, start=1):
            # 每次预测一张，避免一次性加载全部原图。
            result = model.predict(
                source=str(path),
                imgsz=640,
                conf=CONF,
                device=0,
                verbose=False,
            )[0]

            shutil.copy2(path, OUTPUT / "images" / path.name)

            label_path = OUTPUT / "labels" / f"{path.stem}.txt"

            # 先创建空文件，确保没有检测框的图片也有标签文件。
            # 空文件此时仅表示“模型没检测到”，不表示已确认无目标。
            label_path.write_text("", encoding="utf-8")
            result.save_txt(str(label_path), save_conf=False)

            # 预览图显示检测框、类别和置信度。
            result.save(filename=str(OUTPUT / "previews" / path.name))

            boxes = result.boxes
            count = len(boxes)
            min_conf = (
                f"{boxes.conf.min().item():.4f}" if count else ""
            )

            writer.writerow([
                path.name, count, min_conf, "pending", ""
            ])
            file.flush()

            total_boxes += count
            no_detection += int(count == 0)

            print(f"[{index}/{len(images)}] {path.name}：{count} 个候选框")

    print("\n预标注完成")
    print("图片数量：", len(images))
    print("候选框总数：", total_boxes)
    print("没有检测框的图片数量：", no_detection)
    print("输出目录：", OUTPUT)
    print("所有图片仍需人工检查，暂不加入训练。")


if __name__ == "__main__":
    main()