from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "toolkit/data/annotation_work"

images = sorted((WORK / "images").glob("*.jpg"))
assert len(images) == 301, f"图片数量不符：{len(images)}"

for image in images:
    label = WORK / "labels" / f"{image.stem}.txt"
    assert label.is_file(), f"缺少标签：{label}"

# 分成两个压缩包：原始图片、候选标签。
# 使用 x 模式：若已有同名压缩包，停止而不是覆盖。
with ZipFile(WORK / "cvat_images.zip", "x", ZIP_DEFLATED) as archive:
    for image in images:
        archive.write(image, image.name)

with ZipFile(WORK / "cvat_labels.zip", "x", ZIP_DEFLATED) as archive:
    archive.writestr("obj.names", "car\n")
    archive.writestr(
        "obj.data",
        "classes = 1\n"
        "train = train.txt\n"
        "names = obj.names\n"
        "backup = backup/\n",
    )
    archive.writestr(
        "train.txt",
        "".join(f"obj_train_data/{image.name}\n" for image in images),
    )

    for image in images:
        label = WORK / "labels" / f"{image.stem}.txt"
        archive.write(label, f"obj_train_data/{label.name}")

print("打包完成：")
print(WORK / "cvat_images.zip")
print(WORK / "cvat_labels.zip")