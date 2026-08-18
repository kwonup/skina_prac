"""모델 학습 파일.

구현될 학습 흐름:
1. dataset.create_dataloaders()로 train/val batch를 준비한다.
2. model.create_model()로 ResNet18을 만들고 device(CPU/GPU)로 옮긴다.
3. CrossEntropyLoss와 Adam optimizer를 생성한다.
4. 각 epoch에서 model.train() 상태로 batch마다
   forward -> loss 계산 -> zero_grad -> backward -> optimizer.step 순서로 가중치를 갱신한다.
5. epoch가 끝나면 validation.validate()에서 model.eval()과 no_grad()로 검증한다.
6. train/validation loss·accuracy를 W&B에 기록하고, 최고 validation accuracy 모델을 저장한다.

"""
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_model


# 실행 위치와 관계없이 프로젝트 기준 경로로 모델 저장 위치를 만든다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "best_model.pth"
)


def get_device():
    """사용 가능한 가속기를 우선순위대로 선택한다."""

    # NVIDIA GPU가 있는 환경에서는 CUDA를 사용한다.
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon 환경에서는 MPS GPU를 사용한다.
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    # GPU를 사용할 수 없을 때도 CPU로 학습이 가능하도록 한다.
    else:
        return torch.device("cpu")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    # Dropout/BatchNorm 등을 학습 모드로 전환한다.
    model.train()

    # epoch 전체 평균을 구하기 위한 누적 변수다.
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm (
        loader,
        desc="Training",
        leave=False
        ):

        # CPU에 있던 batch를 모델이 있는 device로 옮긴다.
        images = images.to(device)
        labels = labels.to(device)

        # 이전 batch의 gradient 누적을 제거한다.
        optimizer.zero_grad()

        # Forward: 각 이미지의 15개 클래스 logit을 계산한다.
        outputs = model(images)

        # logit과 정답 class index를 비교해 분류 오차를 계산한다.
        loss = criterion(
            outputs,
            labels
        )

        # Backward: 어떤 가중치를 얼마나 바꿔야 할지 gradient를 계산한다.
        loss.backward()

        # 계산한 gradient를 이용해 모델 가중치를 한 번 갱신한다.
        optimizer.step()

        # batch loss에 이미지 수를 곱해 누적한 뒤, 마지막에 전체 이미지 수로 나눈다.
        running_loss += (loss.item()* images.size(0))

        # 가장 큰 logit의 인덱스를 모델의 최종 예측 class로 사용한다.
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    # 모든 batch를 합친 epoch 단위 평균 loss와 accuracy를 만든다.
    epoch_loss = (running_loss / total)

    epoch_accuracy = (correct / total)

    return (epoch_loss,epoch_accuracy)


def validate(
    model,
    loader,
    criterion,
    device
):

    # 검증 중에는 모델의 학습 동작을 끈다.
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # gradient를 계산/저장하지 않으므로 검증은 가중치를 바꾸지 않고 더 적은 메모리를 사용한다.
    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward와 metric 계산은 학습과 같지만 backward/optimizer.step은 수행하지 않는다.
            outputs = model(images)

            loss = criterion( outputs,labels)

            running_loss += (loss.item()* images.size(0))

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = (running_loss / total)

    epoch_accuracy = (correct / total)

    return (epoch_loss,epoch_accuracy)


def main():

    # 첫 End-to-End 점검을 위한 기본 하이퍼파라미터다.
    epochs = 3

    batch_size = 32

    learning_rate = 1e-4

    # 1) 학습에 쓸 장치를 선택한다.
    device = get_device()

    print(
        "Device:",
        device
    )

    # 2) 전처리된 train/validation 데이터를 batch 단위로 불러온다.
    (train_loader,val_loader,_,class_names) = create_dataloaders(batch_size=batch_size)

    print("Classes:",class_names)

    # 3) 폴더에서 읽은 실제 클래스 수에 맞춰 ResNet18의 출력층을 만든다.
    model = create_model(
        num_classes=len(class_names)
    )

    model = model.to(device)

    # 4) 다중 클래스 분류용 loss와 가중치 갱신용 optimizer를 준비한다.
    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # 5) validation 최고 성능 모델을 저장할 폴더가 없으면 생성한다.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 6) 이번 실험의 설정과 epoch별 metric을 W&B에 기록할 run을 시작한다.
    run = wandb.init(
        project="skina_prac",

        name="resnet18_baseline_epoch1",

        config={
            "model": "resnet18",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "image_size": 224,
            "num_classes": len(
                class_names
            )
        }
    )

    # 아직 최고 검증 성능이 없으므로 0에서 시작한다.
    best_val_accuracy = 0.0

    for epoch in range(
        1,
        epochs + 1
    ):

        # 7) train set으로 가중치를 갱신한 뒤 epoch 성능을 계산한다.
        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        # 8) validation set으로는 가중치를 바꾸지 않고 현재 모델을 평가한다.
        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # 9) batch마다가 아니라 epoch당 한 번만 비교 가능한 metric을 남긴다.
        run.log({
            "epoch": epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy
        })

        # 10) validation accuracy가 기존 최고 기록을 넘었을 때만 checkpoint를 갱신한다.
        if (val_accuracy> best_val_accuracy):
            
            best_val_accuracy = (val_accuracy)
            # 가중치뿐 아니라 예측 결과를 해석할 class_names와 성능 정보도 함께 저장한다.
            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        class_names,

                    "val_accuracy":
                        val_accuracy,

                    "epoch":
                        epoch
                },

                MODEL_PATH
            )

            print("Best model saved!")

    print("\nBest Validation Accuracy:",best_val_accuracy)

    # W&B run을 명시적으로 종료해 모든 로그가 전송되도록 한다.
    run.finish()


if __name__ == "__main__":
    main()
