import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/GCC-YOLOv8.yaml')
    model.train(data='ultralytics/cfg/datasets/mydata.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                workers=8,
                device='0',
                optimizer='SGD',
                cfg='ultralytics/cfg/default.yaml',
                project='runs',
                name='train/exp')