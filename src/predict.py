from pathlib import Path
import sys

import torch
from PIL import Image

# 검증 및 추론 전용 전처리(Resize, Normalize 등) 함수를 가져옵니다.
from dataset import eval_transform
# ResNet18 아키텍처 생성 함수를 가져옵니다.
from model import create_model


# 현재 스크립트 위치 기준으로 프로젝트 루트 디렉터리를 탐색합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 추론에 사용할 저장된 최고 성능 가중치 파일 경로입니다.
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"


def get_device():
    """연산 환경에 맞춰 CUDA GPU, Apple Silicon(MPS), CPU 중 최적의 장치를 반환합니다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main(image_path):
    # 1. 연산 장치 확인
    device = get_device()

    # 2. 저장된 체크포인트 파일(가중치 + 메타데이터) 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 체크포인트에 함께 저장해둔 클래스 이름 목록 복원 (예: 15개 피부 질환 클래스명)
    class_names = checkpoint["class_names"]

    # 3. 클래스 개수에 맞추어 모델 구조를 생성하고 학습된 가중치를 주입
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 4. 모델을 추론 전용 모드로 설정 (Dropout, BatchNorm 동작 고정)
    model.eval()

    # 5. 이미지 로드: RGBA나 흑백 이미지일 경우를 대비해 3채널 RGB로 표준화
    image = Image.open(image_path).convert("RGB")

    # 6. 모델 입력 규격에 맞게 전처리 수행 (PIL Image -> Tensor [C, H, W], 보통 [3, 224, 224])
    image_tensor = eval_transform(image)
    print("Transform 후:", image_tensor.shape)

    # 7. 배치(Batch) 차원 추가: 모델은 [B, C, H, W] 형태를 요구하므로 맨 앞에 차원 1개 추가 -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    print("Batch 추가 후:", image_tensor.shape)

    # 8. 역전파 그래프를 생성하지 않고 순전파 추론 수행 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        # 모델의 로짓(Logit) 점수 출력 [1, num_classes]
        outputs = model(image_tensor)

        # Softmax 함수를 적용해 로짓을 0~1 사이의 확률값(합이 1)으로 변환
        probabilities = torch.softmax(outputs, dim=1)

        # 가장 확률이 높은 상위 3개(Top-3)의 확률값과 해당 클래스 인덱스를 추출
        top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

    # 9. 텐서 형태의 결과를 파이썬 기본 리스트(float/int)로 변환
    # [1, 3] 형태이므로 0번 인덱스를 꺼내 1차원 리스트로 변경
    top_probs = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    # 10. 상위 3개 예측 결과 출력
    print("\n=== 예측 결과 ===")
    for rank, (probability, index) in enumerate(zip(top_probs, top_indices), start=1):
        class_name = class_names[index]
        print(f"{rank}. {class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    # 터미널 실행 시 이미지 파일 경로 인자(sys.argv)가 전달되었는지 확인
    if len(sys.argv) != 2:
        print("사용법:\npython src/predict.py 이미지경로")
        sys.exit(1)

    # 첫 번째 인자로 전달받은 이미지 경로를 main 함수에 전달
    main(sys.argv[1])