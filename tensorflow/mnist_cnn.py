import tensorflow as tf
import numpy as np
import os  # 파일 경로 처리를 위해 추가

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
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("------------------------------------------------------------------------------")
print("학습 전 정확도:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")

# 모델 학습
EPOCHS = 1
BATCH_SIZE = 64
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# 모델 저장 코드 추가
# model_save_path = 'saved_models/tensorflow_mnist_cnn'
# # 저장할 디렉토리가 없으면 생성
# os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
# model.save(model_save_path)  # SavedModel 형식으로 저장
# print(f"모델이 {model_save_path}에 저장되었습니다.")

# 모델 불러오기 예시 코드 추가
# 필요할 때 아래 코드를 사용하여 모델을 불러올 수 있습니다
# loaded_model = tf.keras.models.load_model(model_save_path)
# print("저장된 모델을 불러왔습니다.")

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
#     print("------------------------------------------------------------------------------")

# # 예측 결과 출력
# for i, prediction in enumerate(predictions):
#     print(f"Sample {i}: {prediction}")


# 전체 test 데이터에 대한 예측 결과 출력
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("------------------------------------------------------------------------------")
print("학습 후 전체 test 데이터에 대한 예측 결과")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")

# 모델 저장
MODEL_DIRECTORY = 'models'
MODEL_FILENAME = 'mnist_model.h5'
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, MODEL_DIRECTORY, MODEL_FILENAME)

# 모델 파일 이름이 이미 존재하는 경우, 버전 번호를 추가하여 중복을 피합니다.
version = 1
while os.path.exists(model_path):
    MODEL_FILENAME = f'mnist_model_v{version}.h5'
    model_path = os.path.join(script_dir, MODEL_DIRECTORY, MODEL_FILENAME)
    version += 1

# 저장할 디렉토리가 없으면 생성
if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)
model.save(model_path)
print(f"모델이 '{MODEL_FILENAME}'로 저장되었습니다.")