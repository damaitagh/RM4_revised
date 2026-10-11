from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "toolkit/data/rm_dataset_expanded/data.yaml"
WEIGHTS = ROOT / "toolkit/codebase/ultralytics-YOLO26/yolo26n.pt"
def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"找不到数据配置：{DATA}")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"找不到初始权重：{WEIGHTS}")

    model = YOLO(str(WEIGHTS))

    model.train(data=str(DATA),epochs=50,imgsz=960,batch=8,device=0,workers=4,optimizer="auto",seed=42,deterministic=True,mosaic=1.0,close_mosaic=10,mixup=0.0,val=True,save=True,plots=True,project=str(ROOT/"runs"),name="expanded_yolo26n",exist_ok=False,)
    print("训练结果目录：", model.trainer.save_dir)


if __name__ == "__main__":
    main()