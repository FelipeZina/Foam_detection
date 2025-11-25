from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='C:/Users/Ampla Intelligence/Downloads/foam.v1i.yolov8/data.yaml', epochs=20, imgsz=640,batch=14)
