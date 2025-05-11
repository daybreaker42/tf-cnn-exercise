import tensorflow as tf
import numpy as np

# CIFAR-100 데이터셋을 위한 CNN 모델 정의
# CIFAR-100은 32x32 컬러 이미지이므로 입력 형태를 (32, 32, 3)으로 설정합니다.
model = tf.keras.models.Sequential([
    # 첫 번째 컨볼루션 레이어 - 더 깊은 네트워크 구성
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),  # 배치 정규화 추가
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),  # 과적합 방지를 위한 드롭아웃 추가
    
    # 두 번째 컨볼루션 레이어
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    
    # 세 번째 컨볼루션 레이어
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.4),
    
    # 분류를 위한 완전 연결 레이어
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='softmax')  # CIFAR-100은 100개의 클래스가 있습니다
])

# 모델 컴파일 - 학습률 조정 및 최적화 알고리즘 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# CIFAR-100 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
# 정규화 (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# 레이블을 1차원 배열로 변환 (tf.keras.datasets.cifar100.load_data()는 2차원 배열을 반환)
y_train = y_train.reshape(-1).astype('int32')
y_test = y_test.reshape(-1).astype('int32')

# 학습 전 정확도 측정
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("------------------------------------------------------------------------------")
print("학습 전 정확도:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")

# 데이터 증강을 위한 이미지 생성기 설정
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,      # 이미지 회전 범위
    width_shift_range=0.1,  # 가로 이동 범위
    height_shift_range=0.1, # 세로 이동 범위
    horizontal_flip=True,   # 수평 반전 허용
    zoom_range=0.1          # 확대/축소 범위
)
datagen.fit(x_train)

# 콜백 정의 - 조기 종료 및 학습률 감소 설정
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.0001
)

# 모델 학습 - 데이터 증강, 콜백 및 검증 세트 사용
print("모델 학습 시작...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=25,  # 더 많은 에포크로 학습
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# 테스트용 샘플 예측
TEST_NUM = 5  # 예측할 샘플 수
predictions = model.predict(x_test[:TEST_NUM])

# 예측 결과 출력
print("------------------------------------------------------------------------------")
print("예측 샘플 결과:")
for i, prediction in enumerate(predictions):
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    true_class = y_test[i]
    print(f"샘플 {i}:")
    print(f"  실제 클래스: {true_class}")
    print(f"  예측 클래스: {predicted_class}")
    print(f"  정확도: {confidence:.4f}")
    print("  " + ("-" * 50))

# 전체 테스트 데이터에 대한 평가
print("전체 테스트 평가 중...")
loss, accuracy = model.evaluate(x_test, y_test)
print("------------------------------------------------------------------------------")
print("학습 후 전체 테스트 데이터에 대한 예측 결과:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")

# 학습 과정 시각화 코드 (선택적으로 사용 가능)
import matplotlib.pyplot as plt

# 정확도 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cifar100_training_history.png')
print("학습 그래프가 'cifar100_training_history.png'로 저장되었습니다.")
