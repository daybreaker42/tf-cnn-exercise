import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 문제 해결
# Windows에 기본 설치된 맑은 고딕(Malgun Gothic) 폰트 사용
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 현재 파일 위치 확인 및 모델 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(script_dir), 'cifar10_model.h5')

# 저장된 모델 불러오기
print("모델 불러오는 중...")
try:
    model = tf.keras.models.load_model(model_path)
    print(f"모델을 '{model_path}'에서 성공적으로 불러왔습니다!")
    
    # 모델 구조 요약 정보 출력
    print("\n모델 아키텍처:")
    model.summary()
except Exception as e:
    print(f"모델을 불러오는 중 오류 발생: {e}")
    print(f"확인한 모델 경로: {model_path}")
    exit(1)

# CIFAR-10 데이터 로드 및 전처리
print("\nCIFAR-10 테스트 데이터 로드 중...")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 정규화 (0-255 -> 0-1)
x_test = x_test.astype('float32') / 255.0
# 레이블을 1차원 배열로 변환
y_test = y_test.reshape(-1).astype('int32')

# CIFAR-10 클래스 이름 정의
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 불러온 모델의 성능 평가
print("\n불러온 모델 성능 평가 중...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("------------------------------------------------------------------------------")
print("불러온 모델의 테스트 데이터 평가 결과:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")

# 임의의 테스트 이미지에 대한 예측
def predict_random_images(num_samples=5):
    # 랜덤한 인덱스 생성
    random_indices = np.random.choice(len(x_test), size=num_samples, replace=False)
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i, idx in enumerate(random_indices):
        # 이미지 가져오기
        img = x_test[idx]
        true_label = y_test[idx]
        true_class_name = class_names[true_label]
        
        # 예측
        prediction = model.predict(np.expand_dims(img, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]
        confidence = prediction[0][predicted_class]
        
        # 결과 출력
        print(f"\n이미지 {i+1}:")
        print(f"  실제 클래스: {true_label} ({true_class_name})")
        print(f"  예측 클래스: {predicted_class} ({predicted_class_name})")
        print(f"  정확도: {confidence:.4f}")
        
        # 시각화
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        title_color = 'green' if predicted_class == true_label else 'red'
        plt.title(f"예측: {predicted_class_name} ({confidence:.2f})", color=title_color)
        plt.axis('off')
        
        # 예측 확률 바 그래프로 표시
        plt.subplot(num_samples, 2, 2*i+2)
        bars = plt.bar(range(10), prediction[0])
        plt.xticks(range(10), [f"{i}\n{name[:3]}" for i, name in enumerate(class_names)], rotation=0)
        plt.ylabel('확률')
        plt.title('클래스별 예측 확률')
        
        # 실제 클래스와 예측 클래스에 다른 색상 적용
        bars[true_label].set_color('blue')
        if predicted_class != true_label:
            bars[predicted_class].set_color('red')
        else:
            bars[predicted_class].set_color('green')
    plt.tight_layout()
    
    # 결과 저장 경로 설정 (result 폴더)
    result_dir = os.path.join(script_dir, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    result_path = os.path.join(result_dir, 'cifar10_loaded_model_predictions.png')
    plt.savefig(result_path)
    print(f"\n예측 결과 이미지가 '{result_path}'로 저장되었습니다.")
    plt.show()

# 사용자 입력 이미지 예측 함수
def predict_external_image(image_path):
    try:
        # 이미지 불러오기
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 예측
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        predicted_class_name = class_names[predicted_class]
        confidence = prediction[0][predicted_class]
        
        # 결과 출력
        print("\n외부 이미지 예측 결과:")
        print(f"  이미지: {image_path}")
        print(f"  예측 클래스: {predicted_class} ({predicted_class_name})")
        print(f"  정확도: {confidence:.4f}")
        
        # 시각화
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(tf.keras.preprocessing.image.load_img(image_path))
        plt.title(f"원본 이미지")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        bars = plt.bar(range(10), prediction[0])
        bars[predicted_class].set_color('green')
        plt.xticks(range(10), [f"{i}\n{name[:3]}" for i, name in enumerate(class_names)], rotation=0)
        plt.ylabel('확률')
        plt.title('클래스별 예측 확률')
        plt.tight_layout()
        
        # 결과 저장 경로 설정 (result 폴더)
        result_dir = os.path.join(script_dir, 'result')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        result_path = os.path.join(result_dir, 'cifar10_external_prediction.png')
        plt.savefig(result_path)
        print(f"외부 이미지 예측 결과가 '{result_path}'로 저장되었습니다.")
        plt.show()
        
    except Exception as e:
        print(f"외부 이미지 예측 중 오류 발생: {e}")

# 임의의 테스트 이미지 5개에 대한 예측 실행
# print("\n임의의 테스트 이미지에 대한 예측 실행...")
# predict_random_images(5)

# 외부 이미지 경로가 있을 경우 예측 예시
# 주석을 해제하고 유효한 이미지 경로로 변경하여 사용
external_image_path = f'{script_dir}/data/image2.png'  # 예시 이미지 경로
# predict_external_image(external_image_path)

print("\n모델 사용 예시:")
print("1. 임의의 테스트 이미지 예측: predict_random_images(num_samples=5)")
print("2. 외부 이미지 예측: predict_external_image('이미지_파일_경로.jpg')")
print("\n모델을 사용하려면 이 코드를 실행하세요.")
