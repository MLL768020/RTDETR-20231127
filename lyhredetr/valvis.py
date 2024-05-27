import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR,RTDETRyh1

if __name__ == '__main__':

    model = RTDETRyh1('/home/liyihang/lyhredetr/runs/vis/lspp2_4/weights/best.pt')
    # model.load(weights='/home/liyihang/yolov8/runs/train/exp6/weights/best.pt')  # loading pretrain weights
    model.val(data='dataset/VisDrone.yaml',
                split='val',
                imgsz=640,
                epochs=300,
                batch=1,
                workers=1,
                device=7,
                save_json=True,
                # resume=True, # last.pt path

                project='val',
                name='spp',
                )
