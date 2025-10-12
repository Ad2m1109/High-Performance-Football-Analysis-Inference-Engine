import torch
from ultralytics import YOLO

# The YOLO constructor will automatically download yolov8m.pt if it doesn't exist
print("Loading YOLOv8-Medium model (yolov8m.pt)...")
model = YOLO('yolov8m.pt') 

# Export the model to ONNX format. It will be saved as yolov8m.onnx
print("Exporting to yolov8m.onnx...")
model.export(format='onnx', imgsz=640, opset=13)

print("Model successfully exported to yolov8m.onnx")
