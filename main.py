import tensorflow as tf
import numpy as np

# 간단한 CNN 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# MNIST 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# 학습 전 정확도
print("학습 전 정확도:")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")

# 모델 학습
model.fit(x_train, y_train, epochs=1)

# 예측
TEST_NUM = 5 # 예측할 샘플 수
predictions = model.predict(x_test[:TEST_NUM]) # 테스트 데이터에서 TEST_NUM 개의 샘플에 대해 예측을 수행합니다.

# # 예측 결과 출력
# for i, prediction in enumerate(predictions):
#     # 가장 높은 확률을 가진 클래스 선택
#     predicted_label = np.argmax(prediction) # 가장 높은 확률을 가진 클래스의 인덱스를 반환합니다.
#     # 해당 클래스의 확률 값
#     confidence = prediction[predicted_label] # 해당 클래스의 확률 값을 가져옵니다.
#     print(f"Sample {i}:")
#     print(f"  Predicted label: {predicted_label}") # 예측된 레이블(label)을 출력합니다.
#     print(f"  Confidence: {confidence:.4f}") # 해당 레이블의 확률(confidence)을 소수점 네 자리까지 출력합니다.
#     print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")

# # 예측 결과 출력
# for i, prediction in enumerate(predictions):
#     print(f"Sample {i}: {prediction}")

# 전체 test 데이터에 대한 예측 결과 출력
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")
print("전체 test 데이터에 대한 예측 결과")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")