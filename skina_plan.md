# skina — ResNet18 End-to-End 개인 실습 가이드

## 1. 문서 목적

이 문서는 AI-Hub 피부종양 이미지 데이터를 이용하여 **딥러닝 프로젝트 전체 과정을 처음부터 끝까지 직접 경험하기 위한 개인 실습 계획서**이다.

현재 딥러닝 프로젝트 경험이 거의 없는 상태이므로, 처음부터 복잡한 최적화나 서비스 구현을 목표로 하지 않는다.

이번 실습의 핵심 목표는 다음 흐름을 직접 구현하고 이해하는 것이다.

```text
데이터 확인
↓
EDA
↓
Train / Validation / Test 분리
↓
이미지 전처리
↓
Dataset
↓
DataLoader
↓
ResNet18
↓
Loss / Optimizer
↓
Training
↓
Validation
↓
W&B 실험 기록
↓
Best Model 저장
↓
Test
↓
단일 이미지 Inference
```

최종적으로는 각 단계를 직접 설명할 수 있고, 이후 팀 프로젝트에서 다른 모델을 맡더라도 동일한 파이프라인을 스스로 구성할 수 있는 상태를 목표로 한다.

---

# 2. Codex 작업 원칙

Codex는 이 실습을 **한 번에 전부 구현하지 않는다.**

반드시 아래 단계별로 작업한다.

각 단계마다:

1. 필요한 파일만 생성 또는 수정한다.
2. 구현한 내용과 이유를 설명한다.
3. 실행 명령어를 제시한다.
4. 사용자가 직접 실행하여 결과를 확인할 수 있도록 한다.
5. 다음 단계로 넘어가기 전에 현재 단계의 완료 조건을 확인한다.

즉 아래처럼 진행한다.

```text
STEP 1 구현
↓
사용자 실행
↓
결과 확인
↓
문제 수정
↓
STEP 1 완료
↓
STEP 2 진행
```

절대로 처음부터 전체 프로젝트 파일을 한꺼번에 생성하지 않는다.

---

# 3. 데이터

사용 데이터:

- AI-Hub 피부종양 이미지 합성 데이터
- Dataset ID: 71864
- 피부종양 15종 이미지 분류 데이터
- 이미지 크기: 512 × 512
- 총 이미지 수: 13,500장
- 클래스당 900장
- AI-Hub Training 이미지: 클래스당 800장, 총 12,000장
- AI-Hub Validation 이미지: 클래스당 100장, 총 1,500장
- Task: Multi-class Image Classification

이번 개인 실습에서는 **Object Detection이 아니라 Image Classification만 수행한다.**

입력:

```text
피부 이미지
```

출력:

```text
15개 피부종양 클래스 중 하나
```

예:

```text
Input
skin_image.png

↓

ResNet18

↓

악성흑색종
confidence: 0.81
```

---

# 4. 이번 실습에서 사용하는 기술

## Language

```text
Python
```

## Deep Learning

```text
PyTorch
TorchVision
```

## Model

```text
ResNet18
ImageNet Pretrained Weights
```

## Experiment Tracking

```text
Weights & Biases (W&B)
```

## Evaluation

```text
Accuracy
```

1차 실습에서는 Accuracy까지만 필수 구현한다.

전체 파이프라인 성공 후 아래 지표를 추가한다.

```text
Precision
Recall
Macro F1
Confusion Matrix
```

---

# 5. 프로젝트 폴더 구조

초기 실습에서는 구조를 너무 복잡하게 만들지 않는다.

```text
skina_prac/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── validation/
│   │   └── labels/
│   │       ├── train/
│   │       └── validation/
│   │
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── outputs/
│   ├── models/
│   └── plots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 6. 각 폴더 역할

## `data/raw`

AI-Hub에서 받은 데이터를 기존 Training / Validation 구분대로 보관한다.

```text
raw/
├── train/             # AI-Hub Training 이미지, 클래스당 800장
├── validation/        # AI-Hub Validation 이미지, 클래스당 100장
└── labels/            # AI-Hub JSON 라벨 원본 보관
    ├── train/
    └── validation/
```

각 클래스 폴더는 `actinic_keratosis`와 같은 영문 질환명만 사용한다.
상위 폴더가 데이터 용도를 구분하므로 `TS_`, `VS_`, `TL_`, `VL_` 접두사는 사용하지 않는다.

이번 실습은 폴더 이름을 정답으로 사용하는 이미지 분류이므로 JSON 라벨은 학습에 사용하지 않는다.
라벨 파일은 원본 보관 목적으로만 `raw/labels`에 둔다.

이미지와 JSON 파일의 내용은 수정하지 않는다.

---

## `data/processed`

학습용으로 정리한 데이터.

```text
train
validation
test
```

세 데이터셋을 보관한다.

---

## `notebooks`

데이터 확인 및 EDA 용도.

첫 실습에서는:

```text
01_eda.ipynb
```

만 사용한다.

---

## `src`

실제 딥러닝 코드를 작성한다.

---

## `outputs/models`

학습 완료된 PyTorch 모델을 저장한다.

예:

```text
best_model.pth
```

---

## `outputs/plots`

추후 생성하는 그래프를 저장한다.

예:

```text
confusion_matrix.png
training_curve.png
```

---

# 7. 파일별 역할

## `prepare_data.py`

원본 데이터를:

```text
Train
Validation
Test
```

로 나누는 역할.

---

## `dataset.py`

이미지를 PyTorch Dataset으로 읽고 DataLoader를 생성한다.

주요 역할:

```text
ImageFolder
Transform
Dataset
DataLoader
```

---

## `model.py`

Pretrained ResNet18을 불러오고 마지막 FC Layer를 15개 클래스에 맞게 수정한다.

---

## `train.py`

프로젝트의 핵심 파일.

다음을 담당한다.

```text
W&B Run 생성
↓
Model
↓
Loss
↓
Optimizer
↓
Training
↓
Validation
↓
Metric Logging
↓
Best Model 저장
```

---

## `evaluate.py`

저장된 Best Model을 불러와 Test Dataset으로 최종 평가한다.

---

## `predict.py`

새 이미지 한 장을 넣어 실제 예측한다.

---

# 8. 전체 실습 단계

---

# STEP 0 — 개발 환경 구축

## 목표

PyTorch 학습을 실행할 수 있는 환경을 만든다.

필요 패키지:

```text
torch
torchvision
wandb
scikit-learn
matplotlib
pillow
jupyter
```

`requirements.txt`에 기록한다.

예:

```txt
torch
torchvision
wandb
scikit-learn
matplotlib
pillow
jupyter
```

설치:

```bash
pip install -r requirements.txt
```

W&B 로그인:

```bash
wandb login
```

## 완료 조건

- Python 실행 가능
- PyTorch import 가능
- TorchVision import 가능
- W&B 로그인 완료

---

# STEP 1 — 데이터 구조 확인 + EDA

파일:

```text
notebooks/01_eda.ipynb
```

## 목표

모델을 학습하기 전에 데이터가 어떻게 생겼는지 직접 확인한다.

이미지 탐색 대상은 `data/raw/train`과 `data/raw/validation`이다.
`data/raw/labels`의 JSON 파일은 이미지 수 집계와 EDA 대상에서 제외한다.

반드시 확인할 것:

```text
전체 이미지 수
클래스 수
클래스 이름
클래스별 이미지 개수
이미지 크기
RGB 여부
샘플 이미지
```

예상:

```text
전체 이미지 13,500장
클래스 15개
클래스당 900장
이미지 512×512
RGB
```

클래스별 이미지를 여러 장 시각화한다.

예:

```text
악성흑색종

[image] [image] [image]

멜라닌세포모반

[image] [image] [image]
```

## 이해 목표

다음 질문에 답할 수 있어야 한다.

### Q1. 입력 데이터는 무엇인가?

```text
피부 이미지
```

### Q2. Label은 무엇인가?

```text
15개의 피부종양 class 중 하나
```

### Q3. 이 문제는 어떤 문제인가?

```text
Multi-class Image Classification
```

## 완료 조건

- 이미지 한 장 이상 직접 출력
- 클래스 이름 확인
- 클래스별 이미지 개수 확인
- 이미지 크기 확인
- 이미지가 정상적으로 열리는지 확인

---

# STEP 2 — Train / Validation / Test 분리

파일:

```text
src/prepare_data.py
```

## 목표

AI-Hub가 제공한 Training / Validation 구분을 그대로 활용한다.

AI-Hub Training 이미지 12,000장은 모두 Train으로 사용한다.
AI-Hub Validation 이미지 1,500장은 클래스마다 무작위로 50장씩 Validation과 Test에 배정한다.

```text
클래스별
Train      800장
Validation  50장
Test        50장

전체
Train      12,000장
Validation    750장
Test          750장

전체 데이터 기준 비율은 약 88.9% / 5.6% / 5.6%이다.
```

전체 데이터를 다시 합쳐 재분할하지 않는다.
`data/raw/train`은 모두 Train으로 복사하고, `data/raw/validation`만 Validation과 Test로 나눈다.
각 클래스의 Validation / Test 분할 결과가 재실행할 때도 동일하도록 Random Seed를 고정한다.

결과 구조:

```text
data/processed/

├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Dataset 역할 이해

### Train

모델이 실제로 학습하는 데이터.

### Validation

학습 도중 모델 성능을 확인하고 Best Model을 선택하는 데이터.

### Test

모든 학습과 모델 선택이 끝난 후 최종 성능을 확인하는 데이터.

Test 데이터를 보고 하이퍼파라미터를 반복적으로 수정하지 않는다.

## 완료 조건

- train 폴더 생성
- val 폴더 생성
- test 폴더 생성
- 각 폴더에 15개 class 존재
- Train 12,000장 / Val 750장 / Test 750장인지 출력하여 확인
- 각 클래스가 Train 800장 / Val 50장 / Test 50장인지 확인
- 분할 결과가 재실행 시 동일하도록 random seed 사용

추천:

```python
random_state = 42
```

또는

```python
random.seed(42)
```

---

# STEP 3 — Transform / Dataset / DataLoader

파일:

```text
src/dataset.py
```

## 목표

이미지를 PyTorch Model에 입력할 수 있는 형태로 변환한다.

---

## 이미지 전처리

ResNet18 입력용 이미지 크기:

```text
224 × 224
```

Train Transform 예:

```text
Resize
RandomResizedCrop
RandomHorizontalFlip
ToTensor
Normalize
```

Validation / Test Transform:

```text
Resize
ToTensor
Normalize
```

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

---

## Dataset

첫 실습에서는 Custom Dataset을 직접 만들기보다:

```python
torchvision.datasets.ImageFolder
```

를 사용한다.

Dataset 역할:

```text
이미지 한 장
+
해당 이미지의 정답 Label
```

을 반환한다.

---

## DataLoader

Dataset을 Batch 단위로 모델에 전달한다.

초기 설정:

```text
batch_size = 32
```

Train:

```text
shuffle = True
```

Validation / Test:

```text
shuffle = False
```

---

## 반드시 확인

다음 코드와 같은 방식으로 Batch 하나를 꺼내 확인한다.

```python
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
```

예상:

```text
images:
torch.Size([32, 3, 224, 224])

labels:
torch.Size([32])
```

의미:

```text
32 = Batch Size
3 = RGB Channel
224 = Height
224 = Width
```

## 완료 조건

- Train Dataset 생성
- Validation Dataset 생성
- Test Dataset 생성
- DataLoader 생성
- Batch shape 확인
- class names 확인
- 첫 Batch 이미지 시각화 가능

---

# STEP 4 — ResNet18 모델 생성

파일:

```text
src/model.py
```

## 목표

ImageNet으로 Pretrained된 ResNet18을 가져와 피부종양 15종 분류 모델로 변경한다.

구조:

```text
Image

↓

Pretrained ResNet18

↓

FC Layer

↓

15 Classes
```

기존 ResNet18:

```text
1000 classes
```

이번 프로젝트:

```text
15 classes
```

따라서 마지막 Layer를 변경한다.

개념:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    15
)
```

## 초기 학습 방식

첫 번째 실습에서는 구조 단순화를 위해 전체 Fine-Tuning을 허용한다.

즉:

```text
ResNet18 전체 weight 학습
+
새 FC Layer 학습
```

전체 End-to-End 성공 후 다음 실험에서:

```text
Feature Extraction
Freeze
Unfreeze
Fine-Tuning
```

을 비교한다.

## 완료 조건

- ResNet18 다운로드 및 생성 성공
- 마지막 FC Layer 출력이 15인지 확인
- Dummy Batch를 넣었을 때 output shape 확인

예:

```text
Input
[32, 3, 224, 224]

↓

ResNet18

↓

Output
[32, 15]
```

---

# STEP 5 — Loss / Optimizer 이해

파일:

```text
src/train.py
```

## Loss

다중 클래스 분류이므로:

```python
nn.CrossEntropyLoss()
```

사용.

Loss 의미:

```text
모델의 예측이 실제 정답과 얼마나 다른지 나타내는 값
```

---

## Optimizer

첫 실습:

```python
torch.optim.Adam
```

추천 설정:

```text
learning_rate = 1e-4
```

Optimizer 역할:

```text
Loss
↓
Gradient
↓
Weight 수정
```

---

# STEP 6 — W&B 초기 설정

파일:

```text
src/train.py
```

W&B는 프로젝트 시작 단계부터 사용한다.

## W&B 역할

W&B는 이번 실습에서:

```text
학습 실험 기록장
```

역할을 한다.

---

## Project

```text
skina_prac
```

---

## 첫 Run

```text
resnet18-baseline-001
```

---

## Config

반드시 다음 설정을 기록한다.

```text
model
epochs
batch_size
learning_rate
optimizer
image_size
```

예:

```python
run = wandb.init(
    project="skincheck-resnet",
    name="resnet18-baseline-001",
    config={
        "model": "resnet18",
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "image_size": 224,
    }
)
```

첫 번째 실행에서는 반드시:

```text
epochs = 1
```

로 한다.

목표는 성능이 아니라 전체 Pipeline이 정상적으로 실행되는지 확인하는 것이다.

---

# STEP 7 — Training Loop 구현

파일:

```text
src/train.py
```

Training 핵심 흐름:

```text
model.train()

↓

Batch 가져오기

↓

Forward

↓

Loss 계산

↓

optimizer.zero_grad()

↓

loss.backward()

↓

optimizer.step()
```

반드시 직접 구현한다.

핵심 코드 개념:

```python
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

---

## 각 코드의 의미

### `optimizer.zero_grad()`

이전 Batch에서 계산한 Gradient 초기화.

### `model(images)`

Forward Propagation.

이미지를 모델에 넣어 예측 결과 생성.

### `criterion(outputs, labels)`

예측값과 실제 정답 비교.

### `loss.backward()`

Backpropagation.

어떤 Weight를 어떻게 수정해야 하는지 Gradient 계산.

### `optimizer.step()`

계산된 Gradient를 이용해 실제 Weight 수정.

---

# STEP 8 — Training Metric 계산

각 Epoch마다:

```text
train_loss
train_accuracy
```

를 계산한다.

Accuracy:

```text
정답을 맞힌 이미지 수
÷
전체 이미지 수
```

Console에도 출력한다.

예:

```text
Epoch 1/1
Train Loss: 1.8521
Train Accuracy: 0.4512
```

---

# STEP 9 — Validation Loop 구현

Training Epoch가 끝날 때마다 Validation을 수행한다.

Validation에서는 모델 Weight를 수정하면 안 된다.

따라서:

```python
model.eval()
```

사용.

그리고:

```python
with torch.no_grad():
```

안에서 수행한다.

Validation에서는 다음을 실행하지 않는다.

```text
loss.backward()
optimizer.step()
```

계산:

```text
val_loss
val_accuracy
```

예:

```text
Val Loss: 1.4212
Val Accuracy: 0.5821
```

---

# STEP 10 — W&B Metric Logging

각 Epoch 종료 후 다음 값을 W&B에 기록한다.

```text
epoch
train_loss
train_accuracy
val_loss
val_accuracy
```

예:

```python
run.log({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "train_accuracy": train_accuracy,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
})
```

첫 실습에서는 Batch마다 로그하지 않는다.

**Epoch마다 1번만 기록한다.**

---

# STEP 11 — W&B Dashboard 확인

첫 1 Epoch가 정상적으로 끝나면 W&B Dashboard에서 다음을 확인한다.

```text
train_loss
train_accuracy
val_loss
val_accuracy
```

반드시 Run이 생성되었는지 확인한다.

첫 실습의 목표는 좋은 성능이 아니라:

```text
PyTorch Training 결과가
W&B Dashboard까지 정상적으로 기록되는 것
```

이다.

---

# STEP 12 — Best Model 저장

Validation Accuracy를 기준으로 가장 좋은 모델을 저장한다.

저장 위치:

```text
outputs/models/best_model.pth
```

개념:

```python
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy

    torch.save(
        model.state_dict(),
        "outputs/models/best_model.pth"
    )
```

W&B summary에도:

```text
best_val_accuracy
```

를 기록한다.

---

# STEP 13 — 첫 1 Epoch 성공 후 Epoch 증가

1 Epoch End-to-End 성공 후:

```text
epochs = 5
```

로 변경한다.

필요하면:

```text
epochs = 10
```

까지 증가한다.

W&B에서 다음 흐름을 관찰한다.

정상적인 학습 예:

```text
Train Loss ↓
Validation Loss ↓

Train Accuracy ↑
Validation Accuracy ↑
```

과적합 예:

```text
Train Accuracy ↑ 계속 상승

Validation Accuracy
→ 정체 또는 하락
```

이 흐름을 W&B 그래프로 직접 확인한다.

---

# STEP 14 — Test Evaluation

파일:

```text
src/evaluate.py
```

## 목표

Training과 Model Selection이 모두 끝난 뒤 Test Dataset에서 최종 평가한다.

순서:

```text
ResNet18 생성

↓

best_model.pth Load

↓

model.eval()

↓

Test DataLoader

↓

Prediction

↓

Test Accuracy
```

첫 실습에서는:

```text
Test Accuracy
```

를 출력한다.

추가 단계에서:

```text
Precision
Recall
Macro F1
Confusion Matrix
```

를 추가한다.

---

# STEP 15 — Single Image Inference

파일:

```text
src/predict.py
```

## 목표

새 이미지 1장을 넣어 실제 예측 결과를 얻는다.

순서:

```text
Image

↓

PIL Image

↓

Validation Transform

↓

Tensor

↓

Batch Dimension 추가

↓

ResNet18

↓

Softmax

↓

Class Probability
```

원래 한 이미지:

```text
[3, 224, 224]
```

모델 입력:

```text
[1, 3, 224, 224]
```

따라서:

```python
image = image.unsqueeze(0)
```

사용.

---

## 출력

최소:

```text
Predicted Class:
악성흑색종

Confidence:
81.25%
```

가능하면 Top-3도 출력한다.

예:

```text
1. 악성흑색종       81.25%
2. 멜라닌세포모반  11.42%
3. 흑색점            4.13%
```

---

# 9. 1차 End-to-End 완료 조건

다음 모든 항목을 만족하면 첫 실습 완료이다.

- [ ] AI-Hub 데이터 구조 확인
- [ ] 클래스별 샘플 이미지 확인
- [ ] Train / Validation / Test 분리
- [ ] PyTorch Dataset 생성
- [ ] DataLoader 생성
- [ ] Batch Shape 확인
- [ ] ResNet18 Pretrained Model 로드
- [ ] FC Layer를 15 classes로 변경
- [ ] CrossEntropyLoss 적용
- [ ] Adam Optimizer 적용
- [ ] Training Loop 구현
- [ ] Validation Loop 구현
- [ ] W&B Project 생성
- [ ] W&B Run 생성
- [ ] W&B Config 기록
- [ ] train_loss 기록
- [ ] train_accuracy 기록
- [ ] val_loss 기록
- [ ] val_accuracy 기록
- [ ] Best Model 저장
- [ ] Test Accuracy 측정
- [ ] 저장된 Model Load 성공
- [ ] 단일 이미지 Inference 성공
- [ ] Top-1 또는 Top-3 Prediction 출력

---

# 10. 1차 실습에서는 하지 않을 것

첫 End-to-End가 성공하기 전까지 아래 작업은 추가하지 않는다.

```text
FastAPI
Next.js
Grad-CAM
Optuna
Scheduler
Mixed Precision
Docker
Custom CNN 비교
EfficientNet 비교
MobileNet 비교
과도한 Hyperparameter Tuning
```

이유:

```text
전체 딥러닝 Pipeline을 먼저 이해하는 것이
현재 실습의 최우선 목표이기 때문
```

---

# 11. 1차 완료 후 확장 단계

End-to-End 성공 후 아래 순서로 확장한다.

---

## LEVEL 2 — Evaluation 확장

추가:

```text
Precision
Recall
Macro F1
Confusion Matrix
Classification Report
```

---

## LEVEL 3 — Transfer Learning 실험

비교:

```text
ResNet18 전체 Fine-Tuning

vs

Backbone Freeze
FC Layer만 학습

vs

일부 Layer Unfreeze
```

각 실험은 W&B의 다른 Run으로 기록한다.

예:

```text
resnet18-baseline-001

resnet18-freeze-002

resnet18-finetune-003
```

---

## LEVEL 4 — Data Augmentation 실험

비교:

```text
기본 Transform

vs

Data Augmentation 적용
```

W&B에서:

```text
Validation Accuracy
Validation Loss
```

를 비교한다.

---

## LEVEL 5 — W&B 활용 확장

추가로 경험:

```text
run.watch(model)
Gradient 확인
Learning Rate 기록
Prediction Table
Confusion Matrix 기록
```

---

## LEVEL 6 — Grad-CAM

최종 모델이 실제 피부 병변 영역을 보고 판단하는지 확인한다.

---

## LEVEL 7 — 팀 프로젝트

개인 실습 완료 후 팀에서는 동일한 공통 데이터 Split과 실험 조건으로 모델을 나눈다.

예:

```text
Custom CNN
ResNet18
EfficientNet-B0
MobileNetV3
```

공통 기준:

```text
동일 Train / Val / Test Split
동일 image size
동일 metric
동일 seed
동일 W&B Project
```

이후 모델별 성능을 비교한다.

---

# 12. W&B Run 이름 규칙

의미 없는 이름 사용 금지.

금지 예:

```text
test
test2
final
final2
real_final
```

권장:

```text
resnet18-baseline-001

resnet18-lr1e3-002

resnet18-augmentation-003

resnet18-freeze-004

resnet18-finetune-005
```

Run 이름만 보고 어떤 실험인지 이해할 수 있어야 한다.

---

# 13. 공통 하이퍼파라미터 초기값

첫 실습 기준:

```yaml
seed: 42

model: resnet18
pretrained: true

image_size: 224
batch_size: 32

epochs: 1

optimizer: Adam
learning_rate: 0.0001

loss: CrossEntropyLoss

train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15
```

1 Epoch 성공 이후:

```yaml
epochs: 5
```

로 변경한다.

---

# 14. 반드시 학습하면서 이해해야 할 개념

## Dataset

```text
이미지와 Label을 관리하는 객체
```

---

## DataLoader

```text
Dataset에서 데이터를 Batch 단위로 가져오는 객체
```

---

## Batch

```text
모델이 한 번에 처리하는 이미지 묶음
```

예:

```text
batch_size = 32
```

---

## Epoch

```text
전체 Train Dataset을 한 번 모두 학습한 상태
```

---

## Model

```text
이미지로부터 특징을 추출하고 class를 예측
```

---

## Forward Propagation

```text
Image
↓
Model
↓
Prediction
```

---

## Loss

```text
Prediction과 실제 Label이 얼마나 다른지 나타내는 값
```

---

## Backpropagation

```text
Loss를 기준으로 어떤 Weight가 얼마나 영향을 주었는지 계산
```

---

## Optimizer

```text
Gradient를 이용하여 실제 Weight를 수정
```

---

## Learning Rate

```text
한 번에 Weight를 얼마나 크게 수정할지 결정
```

---

## Validation

```text
학습하지 않은 데이터로 현재 Model의 일반화 성능 확인
```

---

## Test

```text
최종 Model의 성능을 마지막에 평가
```

---

# 15. 딥러닝 학습의 핵심 흐름

이번 실습이 끝난 후 아래 내용을 직접 설명할 수 있어야 한다.

```text
Image

↓

Dataset

↓

DataLoader

↓

Batch

↓

ResNet18

↓

Prediction

↓

Label과 비교

↓

CrossEntropyLoss

↓

Backward

↓

Gradient

↓

Adam

↓

Weight Update

↓

다음 Batch

↓

전체 Dataset 반복

↓

1 Epoch 완료

↓

Validation

↓

W&B Logging

↓

Best Model Save
```

---

# 16. Codex에게 요구하는 설명 방식

Codex는 코드를 작성할 때 단순히 완성된 코드만 제공하지 않는다.

각 단계마다 중요한 코드에 대해 다음 수준으로 설명한다.

예:

```python
optimizer.zero_grad()
```

설명:

```text
이전 Batch에서 계산된 Gradient가 PyTorch에서 누적될 수 있으므로,
현재 Batch의 Backpropagation 전에 기존 Gradient를 초기화한다.
```

---

```python
outputs = model(images)
```

설명:

```text
현재 Batch 이미지를 ResNet18에 전달하여
각 이미지에 대한 15개 class score를 계산한다.
```

---

```python
loss.backward()
```

설명:

```text
현재 Loss를 기준으로 각 Weight가 Loss에 얼마나 영향을 주었는지
Gradient를 계산한다.
```

---

```python
optimizer.step()
```

설명:

```text
계산된 Gradient를 이용해 실제 Model Weight를 수정한다.
```

이런 방식으로 **처음 딥러닝 프로젝트를 경험하는 사람이 이해할 수 있도록 설명한다.**

---

# 17. Codex 작업 시 주의사항

1. 한 번에 전체 프로젝트를 구현하지 않는다.
2. 현재 단계에서 필요하지 않은 라이브러리를 추가하지 않는다.
3. 과도한 abstraction을 만들지 않는다.
4. 처음에는 class, factory, config framework 등을 불필요하게 복잡하게 만들지 않는다.
5. 코드의 가독성을 학습 성능보다 우선한다.
6. 함수는 역할별로 명확하게 나눈다.
7. 하이퍼파라미터는 처음에는 쉽게 찾을 수 있는 위치에 둔다.
8. Error가 발생했을 때 임시방편으로 넘어가지 않고 원인을 설명한다.
9. 사용자가 직접 실행할 명령어를 항상 제공한다.
10. 각 STEP의 완료 조건을 충족한 뒤 다음 단계로 이동한다.
11. 데이터 원본을 직접 수정하지 않는다.
12. Test Dataset은 최종 평가 이전까지 모델 선택에 사용하지 않는다.
13. W&B Run은 실험마다 새로 생성한다.
14. Random Seed를 고정하여 Split 재현성을 유지한다.
15. Model 저장 경로와 Dataset 경로를 하드코딩할 경우 프로젝트 루트 기준으로 일관되게 관리한다.

---

# 18. 최종 목표

이번 개인 실습이 끝났을 때 다음 질문에 직접 답할 수 있어야 한다.

```text
왜 Train / Validation / Test를 나누는가?

Dataset과 DataLoader의 차이는 무엇인가?

Batch Size는 무엇인가?

Image를 왜 Tensor로 변환하는가?

ResNet18의 마지막 FC Layer는 왜 바꾸는가?

Pretrained Model이란 무엇인가?

Transfer Learning이란 무엇인가?

CrossEntropyLoss는 무엇을 계산하는가?

optimizer.zero_grad()를 왜 하는가?

loss.backward()에서 무슨 일이 일어나는가?

optimizer.step()은 무엇을 하는가?

model.train()과 model.eval()은 왜 나누는가?

torch.no_grad()는 왜 Validation에서 사용하는가?

Epoch는 무엇인가?

Validation Accuracy는 왜 필요한가?

Best Model은 어떤 기준으로 저장하는가?

Test Dataset은 왜 마지막에 사용하는가?

W&B Project / Run / Config / Log의 차이는 무엇인가?

W&B 그래프를 보고 Overfitting을 어떻게 판단하는가?

저장된 best_model.pth를 어떻게 다시 불러오는가?

새 이미지 한 장을 어떻게 Model에 넣고 예측하는가?
```

이 질문들에 답할 수 있고 실제 코드도 실행할 수 있다면 이번 개인 실습의 목적은 달성된 것이다.

---

# 19. Codex 첫 작업 요청

이 문서를 읽은 뒤 바로 전체 구현을 시작하지 않는다.

첫 번째 작업은 오직 다음 범위만 진행한다.

```text
STEP 0
프로젝트 기본 폴더 구조 생성
requirements.txt
.gitignore

+

STEP 1
notebooks/01_eda.ipynb
```

STEP 1에서는:

```text
데이터 경로 확인
이미지 파일 탐색
전체 이미지 개수 확인
클래스 이름 확인
클래스별 이미지 개수 확인
이미지 크기 확인
샘플 이미지 시각화
```

까지만 구현한다.

ResNet, DataLoader, W&B Training 코드는 아직 구현하지 않는다.

STEP 1 완료 후 사용자가 결과를 확인한 뒤 다음 STEP으로 진행한다.
