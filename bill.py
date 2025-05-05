import os

import torch
from torch.utils.data import Dataset, DataLoader    ## Pytorch의 데이터셋 관련 
from torchvision import transforms                  ## 이미지 데이터 전처리 위한 모듈 

import matplotlib.pyplot as plt                     ## 시각화
import matplotlib.patches as patches                ## 이미지 상에 추가 그래프 관련 
from PIL import Image                               ## 이미지 데이터 처리 모듈 
import numpy as np            

import shutil

# 삭제할 라벨 번호 목록
labels_to_delete = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37,
    62, 63, 64, 65, 66,
    75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85,
    86, 87, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98,
    99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123,
    129, 130, 131, 132,
    133, 134, 135, 136, 137, 138, 139,
    149, 150, 151, 152, 153, 154, 155, 156,
    157, 158, 159, 160, 161, 162, 163,
    164, 165, 166, 167, 168, 169, 170,
    185, 186, 187, 188, 189, 190,
    191, 192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205,
    206, 207, 208, 209, 210, 211
]

# 폴더가 들어있는 상위 디렉터리
base_path = "./data"  # 여기에 train/validation/test가 있음

# 삭제 대상 디렉터리
sets = ["train", "validation", "test"]

for set_name in sets:
    for label in labels_to_delete:
        folder_path = os.path.join(base_path, set_name, str(label))
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"[삭제 완료] {folder_path}")
        else:
            print(f"[없음] {folder_path}")


import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report

# 1. 데이터 전처리
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
}

# 2. 데이터 로딩
data_dir = './data'
image_datasets = {
    phase: datasets.ImageFolder(os.path.join(data_dir, phase), transform=transform[phase])
    for phase in ['train', 'validation', 'test']
}
dataloaders = {
    phase: DataLoader(image_datasets[phase], batch_size=32, shuffle=True)
    for phase in ['train', 'validation', 'test']
}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

# 3. VGG16 모델 로딩 및 수정
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False  # feature extractor 고정

model.classifier[6] = nn.Linear(4096, num_classes)  # 분류기 수정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. 손실 함수 및 최적화기
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

from tqdm import tqdm  # 🔁 로딩바 표시용

# 5. 학습 단계
for epoch in range(50):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloaders['train'], desc=f"[Epoch {epoch+1}]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloaders['train'])
    print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")
    

# 6. 평가 단계 (Epoch 끝날 때마다 정확도 + 손실 계산)
model.eval()
correct = 0
total = 0
test_loss = 0.0  # 🔹 테스트 손실 누적용 변수

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        loss = criterion(outputs, labels)  # 🔹 손실 계산
        test_loss += loss.item()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

# 🔸 평균 손실 계산
avg_test_loss = test_loss / len(dataloaders['test'])
accuracy = correct / total

# 결과 출력
print(f"✅ [Epoch {epoch+1}] Test Accuracy: {accuracy * 100:.2f}% | Test Loss: {avg_test_loss:.4f}\n")

# 모델 저장 
torch.save(model.state_dict(), f"vgg16_epoch{epoch+1}.pt")

