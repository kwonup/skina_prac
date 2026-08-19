"""ImageNet pretrained ResNet18을 15개 피부종양 분류 모델로 바꾼다.

전체 흐름:
[batch, 3, 224, 224] 이미지 텐서
    -> pretrained ResNet18 특징 추출
    -> 교체한 FC layer
    -> [batch, 15] 클래스별 점수(logit)
"""

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


def create_model(num_classes=15, pretrained=True):
    """ResNet18의 마지막 분류기만 현재 클래스 수에 맞춰 교체한다.

    학습 시에는 ImageNet 사전학습 가중치를 사용하고, 저장된 checkpoint를
    불러오는 추론/Grad-CAM 시에는 다운로드가 필요 없도록 pretrained=False를 쓴다.
    """

    # 학습 시작 시에는 사전학습 가중치를, checkpoint 복원 시에는 빈 구조만 만든다.
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(
        weights=weights
    )

    # 기존 FC layer가 받는 특징 벡터의 크기는 유지한다.
    num_features = model.fc.in_features

    # ImageNet의 1,000개 클래스 출력층을 피부종양 num_classes개 출력층으로 교체한다.
    # 이 출력값은 확률이 아닌 logit이며, train.py의 CrossEntropyLoss에 그대로 전달한다.
    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model


if __name__ == "__main__":
    # 이 파일을 직접 실행하면 모델 구조와 입출력 shape을 빠르게 확인한다.
    model = create_model(
        num_classes=15
    )

    # 실제 DataLoader가 주는 batch와 같은 shape의 가상 입력을 만든다.
    dummy_images = torch.randn(
        32,
        3,
        224,
        224
    )

    # forward 결과는 이미지마다 15개 클래스 점수를 가진다: [32, 15]
    outputs = model(dummy_images)

    print(model.fc)
    print("Output shape:", outputs.shape)
