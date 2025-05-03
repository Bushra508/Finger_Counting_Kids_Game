import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# Load the trained grayscale model
model = load_model("dependencies\\model_30.keras")

# Constants
imgSize = 300
IMG_MODEL_SIZE = 128
offset = 20

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    total_fingers = 0

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Create white canvas
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop hand region
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

            # Convert to grayscale, resize to model input
            imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            imgInput = cv2.resize(imgGray, (IMG_MODEL_SIZE, IMG_MODEL_SIZE))
            imgInput = imgInput / 255.0
            imgInput = np.expand_dims(imgInput, axis=(0, -1))

            # Predict and accumulate
            prediction = model.predict(imgInput)
            predicted_class = np.argmax(prediction)
            total_fingers += predicted_class

            # Show prediction per hand
            cv2.putText(img, f'{predicted_class}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show total fingers
    cv2.putText(img, f'Total Fingers: {total_fingers}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)

    # Display the output
    cv2.imshow("Finger Count - Two Hands", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
