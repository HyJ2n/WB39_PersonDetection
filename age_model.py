import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import optimizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from collections import Counter
import os
from tqdm import tqdm
import random
from preprocess_images import preprocess_images 


# 설정 관련 딕셔너리 (하이퍼파라미터)
CFG = {
    'IMG_SIZE':128,  #이미지 사이즈128
    'EPOCHS':100,    #에포크
    'BATCH_SIZE':16, #배치사이즈
    'SEED':1,        #시드
}

# 장치 설정 (CUDA 사용 가능 여부에 따라 CPU 또는 GPU 선택)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Loading...')

# 시드 설정 (재현 가능성을 위해 시드 고정) / 재현성 보장 , 랜덤 요소 제어 
seed = 1
torch.manual_seed(seed)                     # CPU       
torch.cuda.manual_seed(seed)                # GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True       # 하드웨어 및 입력 크기에 따라 최적의 연산 알고리즘을 선택하게 하여 성능을 향상
np.random.seed(seed)                        # NUMPY


# 데이터셋과 모델 설정
class ageDataset(Dataset):
    # DataSet 초기화
    def __init__(self, image, label, train=True, transform=None):
        self.transform = transform # 이미지 변환(transform) 객체를 저장
        self.img_list = image      # 이미지 데이터를 리스트로 저장
        self.label_list = label    # 레이블 데이터를 리스트로 저장

    # DataSet 길이 반환
    def __len__(self):
        return len(self.img_list)

    # DataSet 특정 샘플 반환 
    def __getitem__(self, idx):
        label = self.label_list[idx]
        img = Image.fromarray(np.uint8(self.img_list[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# custom dataset 사용 transform 
# (학습 데이터 변환)
train_transform = torchvision.transforms.Compose([
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
                    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
                    ])

# (테스트 데이터 변환)
test_transform = torchvision.transforms.Compose([
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
                    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
                    ])

# (좌우반전 변환) : 데이터 증강 
Horizontal_transform=torchvision.transforms.Compose([
                    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
                    transforms.RandomHorizontalFlip(1.0),                           # Horizontal = 좌우반전
                    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
                    ])



# 학습 데이터 경로 설정
train_path = sorted(glob.glob(r"C:\Users\admin\Desktop\Project\Main\Age\test\All-Age-Faces Dataset\original images\*.jpg"))
    
    
# 이미지 전처리 실행 (캐싱)
train_path2, face_list = preprocess_images(train_path)


# 파일명을 이용한 나이 추출 및 레이블 설정
file_name = []
for id in train_path2:
    file_name_with_extension = os.path.basename(id)
    name = file_name_with_extension.split('A')[1]
    age = name.split('.')[0]
    file_name.append(age)
    

label_list = list(map(int, file_name))

# 나이 레이블링 (0: 어린이, 1: 청소년, 2: 중년, 3: 노년) / 출력 데이터를 4로 설정했기 때문에 4 레이블은 빠짐
train_y = []
for age in label_list:
    if age < 15:
        train_y.append(0)
    elif age < 30:
        train_y.append(1)
    elif age < 50:
        train_y.append(2)
    elif age < 70:
        train_y.append(3)
    else:
        train_y.append(4)

# 데이터셋 섞기 및 분할 : 순서가 모델 학습에 영향 미치는 것 방지 
random.Random(19991006).shuffle(face_list)
random.Random(19991006).shuffle(train_path2)
random.Random(19991006).shuffle(train_y)

# 전체 데이터셋 80% 훈련, 20% 검증
train_img_list = face_list[:int(len(face_list)*0.8)]
train_label_list = train_y[:int(len(train_y)*0.8)]
valid_img_list = face_list[int(len(face_list)*0.8):]
valid_label_list = train_y[int(len(train_y)*0.8):]

# 클래스 불균형 조정을 위한 가중치 계산 함수
# ( 20~30대의 데이터가 압도적으로 많음 )
def make_weights(labels, nclasses):
    labels = np.array(labels)
    weight_arr = np.zeros_like(labels)

    _, counts = np.unique(labels, return_counts=True)
    for cls in range(nclasses):
        # 클래스의 빈도수에 반비례하여 가중치가 설정
        # ex) 클래스 0의 경우 1/counts[0], 클래스 1의 경우 1/counts[1]
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 

    # 각 데이터 포인트에 대한 가중치가 설정된 배열 : 불균형 데이터셋에서도 공정하게 학습
    return weight_arr

# 클래스 불균형을 해결하기 위한 가중치 계산 함수 호출
weights = make_weights(train_label_list, 4)
weights = torch.DoubleTensor(weights)
weights1 = make_weights(valid_label_list, 4)
weights1 = torch.DoubleTensor(weights1)

# WeightedRandomSampler를 사용한 데이터로더 생성
# WeightedRandomSampler : PyTorch 데이터 샘플링 도구 , DataSet 불균형 경우에 주로 사용
# 각 샘플에 가중치를 부여하여 샘플링 확률을 조정 , 빈도가 낮은 클래스의 샘플들이 학습 과정에서 더 자주 선택되도록 하여 클래스 불균형 문제를 완화
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
sampler1 = torch.utils.data.sampler.WeightedRandomSampler(weights1, len(weights1))

# 학습 DataSet과 검증 DataSet 생성 
train_dataset = ageDataset(image=train_img_list, label=train_label_list, train=True, transform=train_transform)
valid_dataset = ageDataset(image=valid_img_list, label=valid_label_list, train=False, transform=test_transform)

# 학습 DataLoader와 검증 DataLoader 생성 
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=2, sampler=sampler1)

# ResNet 기반 나이 예측 모델 정의
class ResNetAgeModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetAgeModel, self).__init__()
         # 사전 학습된 ResNet-18 모델
        resnet = torchvision.models.resnet18(pretrained=True)
        # ResNet의 마지막 완전 연결 계층(fully connected layer)를 변경
        # 원래 ResNet-18의 마지막 층은 1000개의 클래스를 예측 => 현재 프로젝트는 4개의 나이 그룹을 예측해야 함
        resnet.fc = nn.Sequential(
            nn.Linear(512, 512),            # 512개의 입력 노드와 512개의 출력 노드를 가지는 완전 연결 계층
            nn.ReLU(),                      # ReLU 활성화 함수
            # 과적합 : 모델이 훈련 데이터에 너무 치중하여 새로운 데이터에 대한 예측 성능이 떨어지는 현상 / 드롭아웃 : 이를 방지하기 위한 기법
            nn.Dropout(0.3),                # 과적합을 방지하기 위한 드롭아웃 레이어, 드롭아웃 확률 30%
            nn.Linear(512, num_classes),    # 512개의 입력 노드와 4개의 출력 노드를 가지는 완전 연결 계층
            nn.Softmax(dim=1)               # 각 클래스에 대한 확률을 계산 소프트맥스 함수
        )
        # 변경된 ResNet 모델을 클래스의 속성으로 설정합니다.
        self.resnet = resnet

    # 순전파(forward) 과정
    def forward(self, x):
        # 입력 x를 ResNet 모델에 통과시켜 출력값을 반환
        return self.resnet(x)
    
# 모델 인스턴스 생성 및 장치 설정
model1 = ResNetAgeModel(num_classes=4)
model1.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = None