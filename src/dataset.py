"""학습에 사용할 이미지 폴더를 Dataset과 DataLoader로 연결한다.

전체 흐름:
processed/{train,val,test}/클래스명/이미지
    -> ImageFolder가 폴더명을 숫자 라벨로 변환
    -> transform이 이미지를 모델 입력 텐서로 변환
    -> DataLoader가 텐서를 batch 단위로 전달
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# src/ 기준으로 프로젝트 최상위 폴더를 찾아, 어느 위치에서 실행해도
# 데이터 경로가 일관되도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"


# ImageNet pretrained ResNet18이 학습될 때 사용한 정규화 통계값이다.
# 입력 분포를 맞춰 pretrained weight를 안정적으로 활용한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 학습 데이터에는 매 epoch 조금씩 다른 변형을 적용한다.
# 모델이 특정 위치나 좌우 방향에 과도하게 의존하는 것을 줄이는 증강 단계다.
train_transform = transforms.Compose([
    # 원본에서 임의 영역을 잘라 224x224로 만들며, 입력 크기를 ResNet18에 맞춘다.
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    # 피부 이미지를 좌우 반전해 학습 데이터의 다양성을 늘린다.
    transforms.RandomHorizontalFlip(
        p=0.5
    ),
    # PIL Image(H, W, C)를 PyTorch Tensor(C, H, W)와 0~1 범위로 변환한다.
    transforms.ToTensor(),
    # 각 RGB 채널을 ImageNet 기준으로 정규화한다.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# 검증/테스트는 매번 같은 입력을 사용해야 공정하게 성능을 비교할 수 있으므로
# 랜덤 증강 없이 resize와 정규화만 적용한다.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def create_dataloaders(batch_size=32):
    """세 데이터셋과 DataLoader, 그리고 클래스명 순서를 함께 반환한다."""

    # ImageFolder는 하위 폴더 이름을 클래스명으로 읽고, 알파벳순 인덱스를 부여한다.
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=eval_transform
    )

    # train은 epoch마다 데이터 순서를 섞어 모델이 순서에 의존하지 않게 한다.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # validation/test는 결과 재현과 예측 순서 확인을 위해 섞지 않는다.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # class_names의 인덱스는 labels와 모델 출력 열의 의미를 해석할 때 사용한다.
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )


if __name__ == "__main__":
    # 이 파일을 직접 실행하면 학습 전에 데이터 파이프라인이 정상인지 점검한다.
    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders()
    )

    print("Classes:")
    print(class_names)

    print("Number of classes:", len(class_names))

    # DataLoader에서 batch 하나를 꺼내 모델 입력 형태를 확인한다.
    # 예상: images=[batch_size, 3, 224, 224], labels=[batch_size]
    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("Labels:")
    print(labels)
