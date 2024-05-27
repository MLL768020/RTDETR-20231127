import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/liyihang/yolov8/runs/train/exp6/weights/best.pt')
    model.val(data='dataset/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=1,

              save_json=True,  # if you need to cal coco metrice
              device=6,
              project='runs/val',
              name='ta',
              )
