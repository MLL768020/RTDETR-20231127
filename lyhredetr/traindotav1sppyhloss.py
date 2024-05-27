import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1

if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/fpnupdate.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/dota.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=3,
                #resume='/home/liyihang/lyhredetr/runs/vis/v1yhloss_3/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/dota',
                name='v1yhloss_',
                )
