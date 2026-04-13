from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Load test image
image = cv2.imread("dataset1/test/images/fire2_mp4-37_jpg.rf.742a554b0f0772dc0fb772be8648477c.jpg")

# Run detection
results = model(image)

# Draw bounding boxes
annotated = results[0].plot()

# Show result
cv2.imshow("Fire & Smoke Detection", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()