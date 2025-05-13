"""
CIFAR-10 웹캠 실시간 이미지 분류 프로그램

이 프로그램은 웹캠에서 이미지를 캡처하여 CIFAR-10 데이터셋으로 학습된 
CNN 모델을 사용해 실시간으로 이미지를 분류합니다.
0.5초 간격으로 프레임을 처리하여 시스템 부하를 최소화하고,
디버그 모드 설정을 통해 다양한 수준의 정보를 확인할 수 있습니다.

사용법:
- ESC 키를 눌러 프로그램 종료
- 전역 변수를 수정하여 모델 경로 및 디버그 모드 등을 조정 가능
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 전역 변수 설정 - 사용자가 필요에 따라 조정 가능
MODEL_FILENAME = "cifar10_model.h5"  # 모델 파일명
MODEL_DIRECTORY = "models"  # 모델 디렉토리 (tensorflow 폴더 내의 상대 경로)
DEBUG_MODE = True  # 디버그 모드 (True/False)
CAMERA_INDEX = 0  # 웹캠 인덱스 (내장 웹캠은 보통 0)
FRAME_INTERVAL = 0.1  # 프레임 처리 간격 (초)
DISPLAY_SCALE = 8  # 표시 화면 크기 배율 (CIFAR-10 이미지가 32x32로 작아서 화면에 크게 표시)
WINDOW_SCALE = 2.5  # 표시 윈도우 크기 배율
FONT_PATH = "c:/Windows/Fonts/malgun.ttf"  # 말굴 고딕 폰트 경로

# CIFAR-10 클래스 이름 정의
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_model():
    """모델 파일을 로드합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_DIRECTORY, MODEL_FILENAME)
    
    print(f"모델을 '{model_path}'에서 불러오는 중...")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("모델을 성공적으로 불러왔습니다!")
        if DEBUG_MODE:
            model.summary()
        return model
    except Exception as e:
        print(f"모델을 불러오는 중 오류 발생: {e}")
        print(f"확인한 모델 경로: {model_path}")
        exit(1)

def preprocess_image(image):
    """웹캠에서 캡처한 이미지를 모델 입력에 맞게 전처리합니다."""
    # BGR에서 RGB로 변환 (OpenCV는 BGR 형식, TensorFlow는 RGB 형식 사용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CIFAR-10 크기(32x32)로 리사이즈
    image_resized = cv2.resize(image_rgb, (32, 32))
    
    # 모델 입력 형식에 맞게 정규화 및 차원 확장
    image_normalized = image_resized.astype('float32') / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image_resized

def display_prediction(frame, prediction, image_resized):
    """예측 결과를 화면에 표시합니다."""
    # 예측 결과 추출
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    predicted_label = class_names[predicted_class]
    
    # PIL 이미지로 변환하여 Malgun Gothic 폰트 사용
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # 폰트 로드 (크기는 적절히 조정)
    main_font = ImageFont.truetype(FONT_PATH, 30)
    small_font = ImageFont.truetype(FONT_PATH, 16)
    
    # 결과 텍스트 구성 및 표시
    result_text = f"{predicted_label}: {confidence:.2f}"
    draw.text((10, 30), result_text, font=main_font, fill=(0, 255, 0))
    
    # 디버그 모드일 경우 모든 클래스의 확률 표시
    if DEBUG_MODE:
        y_offset = 70
        for i, class_name in enumerate(class_names):
            prob_text = f"{class_name}: {prediction[0][i]:.4f}"
            color = (0, 255, 0) if i == predicted_class else (255, 0, 0)
            draw.text((10, y_offset), prob_text, font=small_font, fill=color)
            y_offset += 25
            
    # OpenCV 이미지로 다시 변환
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 리사이즈된 이미지(32x32)를 더 크게 표시하기 위해 확대
    display_size = (32 * DISPLAY_SCALE, 32 * DISPLAY_SCALE)
    image_display = cv2.resize(cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), display_size, interpolation=cv2.INTER_NEAREST)
    
    # 확대된 이미지를 원본 프레임의 오른쪽에 표시
    h, w = frame.shape[:2]
    x_offset = w - display_size[0] - 10
    y_offset = 10
    
    # 이미지 영역 확인 및 조정
    if x_offset < 0:
        x_offset = 10  # 공간이 부족하면 왼쪽에 표시
    
    # 입력 이미지 표시
    frame[y_offset:y_offset + display_size[1], x_offset:x_offset + display_size[0]] = image_display
    
    # 이미지 영역 테두리 표시
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + display_size[0], y_offset + display_size[1]), (0, 255, 0), 2)
    
    return frame

def webcam_classifier():
    """웹캠 스트림을 열고 실시간 이미지 분류를 수행합니다."""
    model = load_model()
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 연결을 확인하거나 CAMERA_INDEX 값을 조정하세요.")
        exit(1)
    
    print("\n웹캠 이미지 분류 시작! (종료하려면 ESC 키를 누르세요)")
    last_process_time = 0
    
    # 가장 최근 예측 결과와 이미지를 저장하기 위한 변수
    last_prediction = None
    last_image_resized = None
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 종료합니다.")
            break
        
        current_time = time.time()
        
        # 지정된 간격마다 새로 프레임 처리
        if current_time - last_process_time >= FRAME_INTERVAL:
            # 이미지 전처리
            preprocessed_image, image_resized = preprocess_image(frame)
            
            # 모델 예측
            prediction = model.predict(preprocessed_image, verbose=0)
            
            # 최근 예측 결과와 이미지 저장
            last_prediction = prediction
            last_image_resized = image_resized
            
            # 처리 시간 업데이트
            last_process_time = current_time
        
        # 예측 결과가 있으면 항상 표시 (프레임 처리 간격과 상관없이)
        if last_prediction is not None and last_image_resized is not None:
            frame = display_prediction(frame, last_prediction, last_image_resized)
        
        # 일반 정보 표시
        cv2.putText(frame, f"Debug: {DEBUG_MODE}", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
          # 프레임을 WINDOW_SCALE배 크기로 조정
        display_frame = cv2.resize(frame, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
          # 디스플레이 (영어로 창 제목 설정하여 깨짐 방지)
        cv2.imshow('CIFAR-10 Webcam Image Classification', display_frame)
        
        # ESC 키를 누르면 종료
        if cv2.waitKey(1) == 27:  # ESC 키 코드
            break
    
    # 종료
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    webcam_classifier()
