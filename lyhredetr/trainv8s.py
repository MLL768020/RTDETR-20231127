import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/rt-detr/llf/vis/yolov8-s.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8s.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=1,
                device=0,
                #resume='/home/liyihang/lyhredetr/runs/vis/fpn_10/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/vis',
                name='yolov8s   _',
                )
