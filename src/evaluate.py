from pathlib import Path

import torch

from dataset import create_dataloaders
from model import create_model

# 현재 실행 파일 위치를 기준으로 프로젝트 루트 디렉터리 경로를 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 평가할 최적 가중치 파일(best_model.pth)의 절대 경로를 만듦
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"

def get_device():
    """사용 가능한 가속기(NVIDIA GPU -> Apple Silicon GPU -> CPU)를 우선순위대로 반환."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    # 1. 연산에 사용할 장치(CUDA, MPS, CPU)를 확인하고 지정
    device = get_device()
    print("Device:", device)

    # 2. 테스트 데이터셋만 필요하므로 train/val 로더는 버리고 test_loader만 받음
    _, _, test_loader, _ = create_dataloaders(batch_size=32)

   # 3. 저장된 체크포인트 딕셔너리를 지정한 디바이스 메모리로 로드합
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 4. 저장 시 함께 기록해 둔 클래스 이름 목록을 꺼냄
    class_names = checkpoint["class_names"]

    # 5. 클래스 개수에 맞추어 모델 아키텍처의 출력층(FC Layer)을 생성합
    model = create_model(num_classes=len(class_names))

    # 6. 저장되어 있던 학습된 가중치(weights)를 모델 구조에 주입합니다.
    model.load_state_dict(checkpoint["model_state_dict"])

    # 7. 모델을 연산 장치(GPU/CPU)로 보냄
    model = model.to(device)
    # 8. 모델을 평가 모드로 전환 (Dropout 비활성화, BatchNorm 통계값 고정)
    model.eval()

    # 정확도 계산을 위한 누적 변수 초기화
    correct = 0
    total = 0
    # 9. 평가 시에는 역전파(Backpropagation)가 필요 없으므로 기울기 계산을 꺼서 메모리와 속도를 최적화
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = (outputs.argmax(dim=1))

            correct += ( predictions== labels).sum().item()

            total += labels.size(0)

    # 10. 최종 테스트셋 정확도(Accuracy) 계산
    test_accuracy = (correct / total)

    # 11. 체크포인트 메타데이터와 테스트 최종 결과를 출력
    print( "Best model epoch:",checkpoint["epoch"])
    print("Best validation accuracy:",checkpoint["val_accuracy"])
    print("Test Accuracy:",test_accuracy)

if __name__ == "__main__":
    main()
