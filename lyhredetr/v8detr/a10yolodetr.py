import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':

    model = YOLO('../ultralytics/cfg/models/yolo-detr/15nc/yolov8.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8n.pt')  # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=3,
                # resume=True, # last.pt path
                project='runs/a',
                name='a10yolo',
                )
