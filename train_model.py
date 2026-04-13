from ultralytics import YOLO

# Load base YOLO model
model = YOLO("yolov8n.pt")

# Train the model on fire dataset
model.train(
    data="fire_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)