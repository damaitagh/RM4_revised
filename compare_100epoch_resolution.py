from pathlib import Path
import csv

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "toolkit/data/rm_dataset_expanded/data.yaml"

MODELS = {
    "trained960_100": ROOT / "runs/expanded_yolo26n3/weights/best.pt",
}


def main():
    for path in MODELS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = []

    for model_name, weights in MODELS.items():
        for size in (960, 1280):
            model = YOLO(str(weights))

            # 只验证，不更新模型参数，也不使用测试集。
            metrics = model.val(
                data=str(DATA),
                split="val",
                imgsz=size,
                batch=8,
                device=0,
                workers=4,
                augment=False,
                plots=False,
                project=str(ROOT / "runs/resolution_comparison"),
                name=f"{model_name}_val{size}",
                exist_ok=False,
            )

            rows.append({
                "model": model_name,
                "val_imgsz": size,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            })

    output = ROOT / "resolution_comparison_100epoch.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n验证对比：")
    for row in rows:
        print(
            f"{row['model']} | 输入 {row['val_imgsz']} | "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"mAP50={row['mAP50']:.4f} "
            f"mAP50-95={row['mAP50_95']:.4f}"
        )

    print("\n已保存：", output)


if __name__ == "__main__":
    main()