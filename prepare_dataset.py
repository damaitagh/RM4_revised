from pathlib import Path
import random
import shutil
import json

root = Path("toolkit/data").resolve()
image_dir = root / "labeled/images"
label_dir = root / "labeled/labels"
output = root / "rm_dataset"

# 防止重复运行时覆盖已有划分。
if output.exists():
    raise SystemExit(f"输出目录已存在，停止操作：{output}")

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    p for p in image_dir.iterdir()
    if p.is_file()
    and p.suffix.lower() in extensions
    and p.name != "879.jpg"
)

unlabeled = sorted(
    p for p in (root / "unlabeled/images").iterdir()
    if p.is_file()
    and p.suffix.lower() in extensions
    and p.name != "1643.jpg"
)

assert len(images) == 1510, f"已标注数量不符：{len(images)}"
assert len(unlabeled) == 301, f"未标注数量不符：{len(unlabeled)}"
assert len({p.stem for p in images}) == len(images), "存在重复文件主名"

for image in images:
    assert (label_dir / f"{image.stem}.txt").is_file(), image.name

# 独立随机数生成器：同样的输入和种子得到同样的划分。
rng = random.Random(42)
rng.shuffle(images)

n_train = int(len(images) * 0.8)
n_val = int(len(images) * 0.1)

splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:],
}

# 核对三个集合没有共享文件。
names = {
    split: {p.name for p in files}
    for split, files in splits.items()
}
assert not (names["train"] & names["val"])
assert not (names["train"] & names["test"])
assert not (names["val"] & names["test"])

for split, files in splits.items():
    dst_images = output / "images" / split
    dst_labels = output / "labels" / split
    dst_images.mkdir(parents=True)
    dst_labels.mkdir(parents=True)

    for image in files:
        label = label_dir / f"{image.stem}.txt"
        shutil.copy2(image, dst_images / image.name)
        shutil.copy2(label, dst_labels / label.name)

    print(f"{split}：{len(files)} 张图片，{len(files)} 份标签")

# YOLO 用这个配置文件找到数据集。
yaml_text = (
    f"path: {json.dumps(str(output), ensure_ascii=False)}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: car\n"
)
(output / "data.yaml").write_text(yaml_text, encoding="utf-8")

# 保存划分依据，以及后续允许预标注的图片清单。
manifest = {
    "seed": 42,
    "splits": {
        split: [p.name for p in files]
        for split, files in splits.items()
    },
    "unlabeled_candidates": [
        str(p.relative_to(root)) for p in unlabeled
    ],
    "excluded_duplicates": {
        "labeled/images/879.jpg": "labeled/images/878.jpg",
        "unlabeled/images/1643.jpg": "labeled/images/815.jpg",
    },
}
(output / "split_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n三个集合文件名交集均为 0")
print("后续可用未标注图片：", len(unlabeled))
print("配置文件：", output / "data.yaml")
print("划分记录：", output / "split_manifest.json")