from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import math
import shutil


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "toolkit/data"
BASE = DATA_ROOT / "rm_dataset"
OUTPUT = DATA_ROOT / "rm_dataset_expanded_checked"
LABEL_ZIP = ROOT / "最后一check.zip"
# 本轮暂不使用的空标签图片，不删除原文件。
EXCLUDED = {"1614", "1662", "1705", "1730"}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"输出目录已存在，停止以防覆盖：{OUTPUT}")

    if not LABEL_ZIP.is_file():
        raise FileNotFoundError(f"请先放好标签压缩包：{LABEL_ZIP}")

    manifest = json.loads(
        (BASE / "split_manifest.json").read_text(encoding="utf-8")
    )

    candidates = [
        DATA_ROOT / name
        for name in manifest["unlabeled_candidates"]
    ]

    # 从 ZIP 读取标签文本，不解压覆盖原标签。
    with ZipFile(LABEL_ZIP) as archive:
        names = archive.read("obj.names").decode("utf-8-sig").split()
        if names != ["car"]:
            raise ValueError(f"类别不匹配：{names}")

        labels = {}
        for name in archive.namelist():
            path = Path(name)
            if path.parent.name == "obj_train_data" and path.suffix == ".txt":
                if path.stem in labels:
                    raise ValueError(f"重复标签：{path.stem}")
                labels[path.stem] = archive.read(name).decode("utf-8-sig")

    expected = {path.stem for path in candidates}
    if set(labels) != expected:
        raise ValueError("ZIP 中的标签与301张候选图片不完全对应")

    selected = [p for p in candidates if p.stem not in EXCLUDED]

    # 先检查所有新增样本，再开始复制。
    box_count = 0
    for image in selected:
        if not image.is_file():
            raise FileNotFoundError(image)

        for split in ("train", "val", "test"):
            if (BASE / "images" / split / image.name).exists():
                raise ValueError(f"新增图片与原数据重名：{image.name}")

        lines = [
            line for line in labels[image.stem].splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"发现未处理的空标签：{image.name}")

        for line in lines:
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(map(math.isfinite, values)):
                raise ValueError(f"标签格式错误：{image.name}")

            cls, x, y, w, h = values
            if not (
                cls == 0
                and 0 <= x <= 1 and 0 <= y <= 1
                and 0 < w <= 1 and 0 < h <= 1
            ):
                raise ValueError(f"标签数值错误：{image.name}")

        box_count += len(lines)

    # 原来的三个集合原样复制，验证集、测试集不重新划分。
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            shutil.copytree(
                BASE / kind / split,
                OUTPUT / kind / split,
            )

    # 新增数据只放入 train。
    for image in selected:
        shutil.copy2(image, OUTPUT / "images/train" / image.name)
        destination = OUTPUT / "labels/train" / f"{image.stem}.txt"
        destination.write_text(labels[image.stem], encoding="utf-8")

    config = (
        f"path: {json.dumps(str(OUTPUT), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: car\n"
    )
    (OUTPUT / "data.yaml").write_text(config, encoding="utf-8")

    record = {
        "base_dataset": str(BASE),
        "label_zip": LABEL_ZIP.name,
        "label_zip_sha256": hashlib.sha256(
            LABEL_ZIP.read_bytes()
        ).hexdigest(),
        "added_images": [p.name for p in selected],
        "excluded_images": [f"{stem}.jpg" for stem in sorted(EXCLUDED)],
        "exclusion_reason": "本轮暂不使用空标签候选图片",
        "added_boxes": box_count,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split in ("train", "val", "test"):
        count = len(list((OUTPUT / "images" / split).glob("*.jpg")))
        print(f"{split}：{count} 张")

    print("新增图片：", len(selected))
    print("新增目标框：", box_count)
    print("数据配置：", OUTPUT / "data.yaml")


if __name__ == "__main__":
    main()