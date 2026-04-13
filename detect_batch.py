from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs/detect/train2/weights/best.pt")

input_folder = "dataset1/test/images"
output_folder = "results"

os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(input_folder):

    image_path = os.path.join(input_folder, image_name)

    if image_name.endswith((".jpg", ".png", ".jpeg")):

        image = cv2.imread(image_path)

        results = model(image)

        annotated = results[0].plot()

        save_path = os.path.join(output_folder, image_name)

        cv2.imwrite(save_path, annotated)

        print(f"Processed: {image_name}")

print("All images processed successfully")