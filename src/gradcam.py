"""저장된 피부종양 분류 모델의 Grad-CAM heatmap을 생성한다.

실행 예시:
    python src/gradcam.py data/processed/test/actinic_keratosis/example.png

기본값은 모델이 예측한 클래스를 설명한다. ``--target-class``를 주면 그 클래스의
점수가 높아지는 이미지 영역을 시각화하므로, 오답 분석에도 활용할 수 있다.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import eval_transform
from model import create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "gradcam"


def get_device():
    """CUDA, Apple Silicon MPS, CPU 순으로 사용 가능한 연산 장치를 선택한다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_model(device):
    """best_model.pth에서 모델 가중치와 클래스명 순서를 복원한다."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "저장된 모델이 없습니다. 먼저 train.py를 실행해 "
            f"{MODEL_PATH}를 생성하세요."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    # checkpoint에는 학습된 가중치가 있으므로 ImageNet weight를 다시 받지 않는다.
    model = create_model(
        num_classes=len(class_names),
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def prepare_image(image_path, device):
    """Grad-CAM overlay용 RGB 이미지와 모델 입력 텐서를 함께 만든다."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    # overlay와 모델 입력이 같은 224x224 이미지를 기준으로 하도록 맞춘다.
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb_image = np.asarray(image, dtype=np.float32) / 255.0

    # 학습 중 validation/test에 사용한 것과 동일한 정규화를 적용한다.
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    return image, rgb_image, input_tensor


def get_target_index(model, input_tensor, class_names, target_class):
    """기본 예측 클래스 또는 사용자가 지정한 클래스의 인덱스를 정한다."""
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_index].item()

    if target_class is None:
        return predicted_index, predicted_index, confidence

    if target_class not in class_names:
        available_classes = ", ".join(class_names)
        raise ValueError(
            f"알 수 없는 클래스입니다: {target_class}\n"
            f"사용 가능한 클래스: {available_classes}"
        )

    return class_names.index(target_class), predicted_index, confidence


def create_gradcam(image_path, target_class=None, output_path=None):
    """이미지 한 장의 Grad-CAM overlay를 만들고 결과 경로를 반환한다."""
    device = get_device()
    model, class_names, checkpoint = load_trained_model(device)
    _, rgb_image, input_tensor = prepare_image(image_path, device)

    target_index, predicted_index, confidence = get_target_index(
        model,
        input_tensor,
        class_names,
        target_class
    )

    # ResNet18의 마지막 convolution block은 공간 정보가 남아 있어 Grad-CAM target으로 적합하다.
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(target_index)]

    # Grad-CAM은 target class 점수에 대한 gradient와 해당 layer의 activation을 결합한다.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    overlay = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"gradcam_{Path(image_path).stem}.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Best validation accuracy:", checkpoint["val_accuracy"])
    print("Predicted class:", class_names[predicted_index])
    print(f"Prediction confidence: {confidence * 100:.2f}%")
    print("Grad-CAM target:", class_names[target_index])
    print("Saved:", output_path)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="저장된 ResNet18 모델로 이미지 한 장의 Grad-CAM을 생성합니다."
    )
    parser.add_argument("image_path", help="Grad-CAM을 생성할 이미지 경로")
    parser.add_argument(
        "--target-class",
        help="설명할 클래스명. 생략하면 모델의 예측 클래스를 사용합니다."
    )
    parser.add_argument(
        "--output",
        help="결과 PNG 경로. 생략하면 outputs/plots/gradcam/에 저장합니다."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_gradcam(
        image_path=args.image_path,
        target_class=args.target_class,
        output_path=args.output
    )
