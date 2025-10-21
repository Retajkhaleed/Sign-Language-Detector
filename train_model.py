import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# تحميل الصور والملصقات
data, labels = [], []
classes = sorted([d for d in os.listdir('dataset/asl data') if os.path.isdir(f'dataset/asl data/{d}')])

# حفظ أسماء الحروف
os.makedirs('model', exist_ok=True)
with open('model/labels.txt', 'w') as f:
    for label in classes:
        f.write(label + '\n')

# قراءة الصور وتحويلها إلى رمادية ومقاس موحد
for idx, gesture in enumerate(classes):
    folder = f'dataset/asl data/{gesture}'
    for img_name in os.listdir(folder)[:300]:
        if img_name.lower().endswith('.png'):
            path = f'{folder}/{img_name}'
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                data.append(img)
                labels.append(idx)


# تجهيز البيانات للتدريب
X = np.array(data).reshape(-1, 64, 64, 1) / 255.0
y = to_categorical(labels)

# بناء النموذج
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# تدريب النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=32)

# حفظ النموذج
model.save('model/asl_model.h5')
