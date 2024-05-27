import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/v8rtdetrecm.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8s.pt')  # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=5,
                resume=True, # last.pt path
                project='runs/llf',
                name='yoloredetrecm',
                )