from ultralytics import YOLO
import cv2
import os

# Load trained YOLO model (pretrained for now)
model = YOLO("yolov8n.pt")

# Image path
image_path = "test_images/normal_image.jpg"

if not os.path.exists(image_path):
    print("Image not found")
    exit()

# Read image
image = cv2.imread(image_path)

print("Running detection...")

# Run model
results = model(image)

# Draw bounding boxes
annotated_frame = results[0].plot()

# Save result
output_path = "results/output.jpg"
cv2.imwrite(output_path, annotated_frame)

# Show result
cv2.imshow("Fire & Smoke Detection", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection complete")