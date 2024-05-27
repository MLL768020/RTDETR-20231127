import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR
# RTDETR giou
if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/vis/v8rtdetr/yuan.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=1,
                device=0,
                resume ='/home/liyihang/lyhredetr/runs/vis/yuan_3/weights/last.pt',
                project='/home/liyihang/lyhredetr/runs/vis',
                name='yuan_',
                )
