import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/vis/v8256decodersppd.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=2,
                workers=4,
                device=3,
                resume='/home/liyihang/lyhredetr/runs/vis/v8256decodersppd_/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/vis',
                name='v8256decodersppd_',
                )
