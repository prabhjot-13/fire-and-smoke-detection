import cv2
import numpy as np

print("Program started")

# create a simple black image
img = np.zeros((400,600,3), dtype=np.uint8)

# write text on it
cv2.putText(img, "Fire & Smoke Detection Prototype", (40,200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Test Window", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Program finished")