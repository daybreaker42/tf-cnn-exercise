# TensorFlow를 이용한 CNN 예제

이 프로젝트는 TensorFlow와 Keras API를 사용하여 MNIST 손글씨 데이터셋과 CIFAR-10 컬러 이미지 데이터셋에 대한 CNN(Convolutional Neural Network) 모델을 구현한 예제입니다.

## 프로젝트 개요

이 예제는 다음과 같은 과정으로 구성됩니다:
- 다양한 복잡도의 CNN 모델 구축
- MNIST 및 CIFAR-10 데이터셋 로드 및 전처리
- 모델 학습 및 저장
- 모델 평가 및 예측 수행
- 웹캠을 이용한 실시간 이미지 분류 기능 (CIFAR-10)

## 기술 스택

- Python 3.11
- TensorFlow 2.19.0
- NumPy
- OpenCV (웹캠 기능)
- Matplotlib (시각화)

## 환경 설정 및 설치 방법

### 1. Python 3.11 가상 환경 설정

#### Windows에서 설정
```bash
# 가상 환경 생성
python -m venv venv_cnn_py311 --python=3.11

# 가상 환경 활성화
venv_cnn_py311\Scripts\activate
```

#### macOS/Linux에서 설정
```bash
# 가상 환경 생성
python3.11 -m venv venv_cnn_py311

# 가상 환경 활성화
source venv_cnn_py311/bin/activate
```

### 2. 필요한 라이브러리 설치
가상 환경을 활성화한 후 다음 명령어를 실행하여 필요한 라이브러리를 설치합니다:
```bash
pip install tensorflow==2.19.0 numpy matplotlib opencv-python pillow
```
> 참고: Apple Silicon(M1/M2) Mac 사용자는 `tensorflow-macos`를 설치해야 할 수 있습니다.

## 모델 아키텍처

### MNIST 모델 구조
간단한 CNN 모델로 구성됩니다:
1. 컨볼루션 레이어 (32개의 3x3 필터, ReLU 활성화 함수)
2. 맥스 풀링 레이어 (2x2)
3. Flatten 레이어
4. 완전 연결 레이어 (10개의 출력 뉴런, softmax 활성화 함수)

### CIFAR-10 모델 구조
더 복잡한 CNN 모델로 구성됩니다:
1. 첫 번째 컨볼루션 블록 (64개의 3x3 필터, BatchNormalization, MaxPooling, Dropout)
2. 두 번째 컨볼루션 블록 (128개의 3x3 필터, BatchNormalization, MaxPooling, Dropout)
3. 세 번째 컨볼루션 블록 (256개의 3x3 필터, BatchNormalization, MaxPooling, Dropout)
4. Flatten 레이어
5. 완전 연결 레이어 (512 뉴런, BatchNormalization, Dropout)
6. 출력 레이어 (10개의 출력 뉴런, softmax 활성화 함수)

## 데이터셋

본 프로젝트에서는 Keras API를 통해 직접 데이터셋을 로드하여 사용합니다:

```python
# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

### MNIST 데이터셋
Keras에서 제공하는 손글씨 숫자 데이터셋:
- 60,000개의 학습 이미지
- 10,000개의 테스트 이미지
- 28x28 픽셀 그레이스케일 이미지
- 0부터 9까지의 10개 클래스
- 데이터 형태: (samples, 28, 28) - 그레이스케일 이미지
- 자동으로 다운로드 및 캐시: `~/.keras/datasets/mnist.npz`에 저장

### CIFAR-10 데이터셋
Keras에서 제공하는 컬러 이미지 분류 데이터셋:
- 50,000개의 학습 이미지
- 10,000개의 테스트 이미지
- 32x32 픽셀 RGB 컬러 이미지
- 10개의 클래스: 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭
- 데이터 형태: (samples, 32, 32, 3) - RGB 컬러 이미지
- 자동으로 다운로드 및 캐시: `~/.keras/datasets/cifar-10-batches-py`에 저장

### 데이터 전처리
두 데이터셋 모두 다음과 같은 전처리 과정을 거칩니다:
1. 픽셀 값 정규화 (0-255 → 0-1)
2. 필요에 따라 차원 확장 (MNIST의 경우 CNN 입력을 위해 채널 차원 추가)
3. CIFAR-10의 경우 레이블을 1차원 배열로 변환

## 프로젝트 파일 구조

- `tensorflow/mnist_cnn.py`: MNIST 데이터셋에 대한 간단한 CNN 모델 구현
- `tensorflow/cifar10_cnn.py`: CIFAR-10 모델 학습 및 저장 스크립트
- `tensorflow/cifar10_load_model.py`: 저장된 CIFAR-10 모델을 로드하여 테스트하는 스크립트
- `tensorflow/cifar10_webcam.py`: 웹캠을 사용하여 실시간 이미지 분류를 수행하는 스크립트
- `models/`: 학습된 모델이 저장되는 디렉토리
- `result/`: 예측 결과 이미지가 저장되는 디렉토리

## 사용 방법

### MNIST 모델
```bash
# MNIST 모델 학습 및 저장
python tensorflow/mnist_cnn.py
```

### CIFAR-10 모델
```bash
# CIFAR-10 모델 학습 및 저장
python tensorflow/cifar10_cnn.py

# 저장된 CIFAR-10 모델 테스트
python tensorflow/cifar10_load_model.py

# 웹캠을 이용한 CIFAR-10 실시간 분류
python tensorflow/cifar10_webcam.py
```

## 데이터 증강 및 최적화 기법 (CIFAR-10)

CIFAR-10 모델의 성능 향상을 위해 다양한 기법이 적용되었습니다:
- 이미지 회전, 이동, 반전 등의 데이터 증강
- 배치 정규화(Batch Normalization)
- 드롭아웃(Dropout)
- 조기 종료(Early Stopping)
- 학습률 감소(Learning Rate Reduction)

## 모델 성능 비교

| 모델 | 데이터셋 | 정확도 | 에포크 | 복잡도 | 특징 |
|------|---------|-------|-------|--------|------|
| MNIST CNN | MNIST | ~98% | 1 | 낮음 | 간단한 구조, 빠른 학습 속도 |
| CIFAR-10 CNN | CIFAR-10 | ~75-85% | 15 | 높음 | 데이터 증강, 배치 정규화, 드롭아웃 등 적용 |

## 추가 정보

이 프로젝트는 교육 목적으로 만들어진 CNN 모델 예제입니다. MNIST는 간단한 CNN 구조로도 높은 정확도를 달성할 수 있지만, CIFAR-10은 더 복잡한 모델과 다양한 최적화 기법이 필요합니다.

## 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [Keras 데이터셋 API](https://keras.io/api/datasets/)
- [MNIST 데이터셋](https://keras.io/api/datasets/mnist/)
- [CIFAR-10 데이터셋](https://keras.io/api/datasets/cifar10/)
- [CNN 개요](https://cs231n.github.io/convolutional-networks/)
- [배치 정규화 논문](https://arxiv.org/abs/1502.03167)
- [Python 가상 환경 문서](https://docs.python.org/3.11/library/venv.html)

