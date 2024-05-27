import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':
    model = RTDETR('/home/liyihang/lyhredetr/runs/llf/phre/weights/best.pt')
    model.val(data='dataset/pheno.yaml',
              split='val',
              imgsz=640,
              batch=1,

              save_json=True,  # if you need to cal coco metrice
              device=1,
              project='runs/val',
              name='ta',
              )
