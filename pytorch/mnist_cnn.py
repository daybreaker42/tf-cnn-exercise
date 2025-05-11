import torch
import torch.nn as nn
import torch.nn.functional as F  # F 모듈 import 추가
import torch.optim as optim
import numpy as np  # numpy 추가
import os  # 파일 경로 처리를 위해 추가
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 수정: activation 매개변수 제거
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 수정: 입력 크기 계산 (28x28 -> 26x26 conv -> 13x13 pool)
        self.fc1 = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        # 수정: F.relu 함수를 별도로 호출
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        # 수정: softmax 함수 대신 로짓 값 반환 (CrossEntropyLoss가 내부적으로 softmax 적용)
        x = self.fc1(x)
        return x

# 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 초기화
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 전 정확도 평가 추가
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# 학습 전 정확도 출력 추가
loss, accuracy = evaluate_model(model, test_loader)
print("------------------------------------------------------------------------------")
print("학습 전 정확도:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")

# 학습 루프
for epoch in range(1):  # 1 epoch만 실행
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # 학습 중간 로그 출력 (100번째 배치마다)
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 모델 저장 코드 추가
model_save_dir = 'saved_models'
model_save_path = os.path.join(model_save_dir, 'pytorch_mnist_cnn.pt')
# 저장할 디렉토리가 없으면 생성
os.makedirs(model_save_dir, exist_ok=True)

# 모델 저장 - 전체 모델 저장 방식
torch.save(model, model_save_path)
print(f"모델이 {model_save_path}에 저장되었습니다.")

# 모델 저장 - 모델 가중치만 저장하는 방식 (더 권장됨)
weights_save_path = os.path.join(model_save_dir, 'pytorch_mnist_cnn_weights.pt')
torch.save(model.state_dict(), weights_save_path)
print(f"모델 가중치가 {weights_save_path}에 저장되었습니다.")

# 모델 불러오기 예시 코드 추가 (주석 처리)
# 전체 모델 불러오기 방식
# loaded_model = torch.load(model_save_path)
# loaded_model.eval()  # 평가 모드로 설정

# 모델 가중치만 불러오기 방식 (더 권장됨)
# new_model = SimpleCNN()  # 모델 구조를 먼저 정의
# new_model.load_state_dict(torch.load(weights_save_path))
# new_model.eval()  # 평가 모드로 설정
# print("저장된 모델을 불러왔습니다.")

# 예측 추가
TEST_NUM = 5  # 예측할 샘플 수
test_samples = []
test_targets = []

# 테스트 데이터에서 샘플 추출
for data, target in test_loader:
    test_samples.extend(data[:TEST_NUM])
    test_targets.extend(target[:TEST_NUM])
    break  # 첫 번째 배치에서만 가져옴

# 예측 실행
model.eval()
with torch.no_grad():
    predictions = []
    for i in range(TEST_NUM):
        output = model(test_samples[i].unsqueeze(0))
        predictions.append(F.softmax(output, dim=1).numpy()[0])

# 예측 결과 출력 방식 1 (주석처리된 첫 번째 방식과 유사)
# for i, prediction in enumerate(predictions):
#     predicted_label = np.argmax(prediction)
#     confidence = prediction[predicted_label]
#     print(f"Sample {i}:")
#     print(f"  Predicted label: {predicted_label}")
#     print(f"  Confidence: {confidence:.4f}")
#     print("------------------------------------------------------------------------------")

# 예측 결과 출력 방식 2 (주석처리된 두 번째 방식과 유사)
# for i, prediction in enumerate(predictions):
#     print(f"Sample {i}: {prediction}")

# 전체 test 데이터에 대한 예측 결과 출력
loss, accuracy = evaluate_model(model, test_loader)
print("------------------------------------------------------------------------------")
print("학습 후 전체 test 데이터에 대한 예측 결과")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("------------------------------------------------------------------------------")