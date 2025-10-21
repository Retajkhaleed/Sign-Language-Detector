import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/asl_model.h5')

# قراءة ترتيب الحروف من الملف
with open('model/labels.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" لم يتم التقاط صورة من الكاميرا")
        break

    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 64, 64, 1))

    prediction = model.predict(reshaped)
    label = classes[np.argmax(prediction)]

    cv2.putText(frame, f'Gesture: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
