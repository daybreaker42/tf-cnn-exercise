# TensorFlow를 이용한 CNN 예제

이 프로젝트는 TensorFlow와 Keras API를 사용하여 MNIST 손글씨 데이터셋에 대한 간단한 CNN(Convolutional Neural Network) 모델을 구현한 예제입니다.

## 프로젝트 개요

이 예제는 다음과 같은 과정으로 구성됩니다:
- 간단한 CNN 모델 구축
- MNIST 데이터셋 로드 및 전처리
- 모델 학습
- 모델 평가 및 예측 수행

## 기술 스택

- Python 3.11
- TensorFlow 2.19.0
- NumPy

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
pip install tensorflow==2.19.0 numpy
```
> 참고: Apple Silicon(M1/M2) Mac 사용자는 `tensorflow-macos`를 설치해야 할 수 있습니다.

## 모델 아키텍처

이 프로젝트에서 사용된 CNN 모델은 다음과 같은 구조를 가집니다:
1. 컨볼루션 레이어 (32개의 3x3 필터, ReLU 활성화 함수)
2. 맥스 풀링 레이어 (2x2)
3. Flatten 레이어
4. 완전 연결 레이어 (10개의 출력 뉴런, softmax 활성화 함수)

## 데이터셋

MNIST 손글씨 숫자 데이터셋을 사용합니다:
- 60,000개의 학습 이미지
- 10,000개의 테스트 이미지
- 28x28 픽셀 그레이스케일 이미지
- 0부터 9까지의 10개 클래스

## 사용 방법

1. 가상 환경 활성화 후 프로젝트 실행:
```bash
python main.py
```

2. 실행 결과:
- 학습 전 모델의 성능 평가
- 1 에포크 학습 수행
- 테스트 데이터셋에 대한 최종 성능 평가

## 추가 정보

이 예제는 교육 목적으로 만들어진 간단한 CNN 모델입니다. 실제 응용에서는 더 깊은 네트워크, 드롭아웃, 배치 정규화 등 추가적인 기술이 사용될 수 있습니다.

## 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)
- [CNN 개요](https://cs231n.github.io/convolutional-networks/)
- [Python 가상 환경 문서](https://docs.python.org/3.11/library/venv.html)

