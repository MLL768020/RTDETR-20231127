import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r50.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/rtdetr-r50.pt')  # loading pretrain weights
    model.train(data='dataset/pheno.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=1,
                # resume=True, # last.pt path
                project='runs/llf',
                name='phr50',
                )
