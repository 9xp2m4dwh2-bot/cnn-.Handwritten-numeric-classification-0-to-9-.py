import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ===================================
# Mr.pang의 첫 CNN 모델!
# MNIST 손글씨 숫자 분류 (0~9)
# 2026.4.19
# ===================================

# 1. 데이터 로드
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. 전처리 (손질하기)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("X_train shape:", X_train.shape)
print("값 범위:", X_train.min(), "~", X_train.max())

# 3. CNN 모델 만들기
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 4. 컴파일 (설정)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 학습!
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# 6. 테스트
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n최종 정확도: {test_acc*100:.2f}%")
# 첫 번째 Conv2D 층의 필터 가져오기
first_conv = model.layers[0]
filters, biases = first_conv.get_weights()

print("필터 개수:", filters.shape[3])  # 32개
print("필터 크기:", filters.shape[:2])  # 3x3

# 16개 필터 그려보기
plt.figure(figsize=(12, 6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    f = filters[:, :, 0, i]
    plt.imshow(f, cmap='gray')
    plt.axis('off')
    plt.title(f'Filter {i+1}')
plt.suptitle('First Conv2D Filters (3x3)')
plt.show()
