"""AI-Hub 원본 이미지를 학습용 train/validation/test 구조로 준비한다.

전체 흐름:
raw/train(클래스당 800장) -> processed/train에 모두 복사
raw/validation(클래스당 100장) -> seed 고정 후 50장씩 val/test로 분리

원본 raw 데이터는 수정하지 않고 processed 데이터만 새로 만든다.
"""

from pathlib import Path
import random
import shutil


# 실행할 때마다 validation/test에 같은 이미지가 배정되도록 고정한다.
SEED = 42


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
RAW_VALIDATION_DIR = PROJECT_ROOT / "data" / "raw" / "validation"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"


def main():

    # 1) validation 이미지의 무작위 분할 결과를 재현 가능하게 만든다.
    random.seed(SEED)

    # 2) 이전에 만든 processed 결과가 있다면 제거한다.
    #    원본(raw)은 건드리지 않으며, 이후 현재 규칙으로 전체를 다시 만든다.
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    # 3) ImageFolder가 읽을 train/val/test/클래스명 구조의 상위 폴더를 만든다.
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 4) train 원본의 하위 폴더 이름을 클래스 라벨 목록으로 사용한다.
    class_names = sorted([
        folder.name
        for folder in RAW_TRAIN_DIR.iterdir()
        if folder.is_dir()
    ])

    print("클래스 목록:")
    print(class_names)

    print("클래스 수:", len(class_names))

    # 5) AI-Hub가 제공한 train 800장/클래스는 재분할하지 않고 모두 train으로 복사한다.

    for class_name in class_names:

        src_dir = RAW_TRAIN_DIR / class_name
        dst_dir = TRAIN_DIR / class_name

        shutil.copytree(
            src_dir,
            dst_dir
        )

    print("Train 데이터 복사 완료")

    # 6) AI-Hub validation 100장/클래스만 val 50장, test 50장으로 나눈다.

    for class_name in class_names:

        validation_class_dir = RAW_VALIDATION_DIR / class_name

        image_files = [
            path
            for path in validation_class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if len(image_files) != 100:
            raise ValueError(
                f"{class_name}: "
                f"validation 이미지가 100장이 아닙니다. "
                f"현재 {len(image_files)}장"
            )

        # 파일 목록을 섞은 뒤 앞/뒤 50장을 나누므로 seed가 같으면 결과도 같다.
        random.shuffle(image_files)

        val_files = image_files[:50]
        test_files = image_files[50:]

        val_class_dir = VAL_DIR / class_name
        test_class_dir = TEST_DIR / class_name

        val_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        test_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # 7) 선택된 파일을 복사한다. copy2는 원본의 수정 시간 등 메타데이터도 보존한다.
        for image_path in val_files:

            shutil.copy2(
                image_path,
                val_class_dir / image_path.name
            )

        for image_path in test_files:

            shutil.copy2(
                image_path,
                test_class_dir / image_path.name
            )

        print(
            f"{class_name}: "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

    # 이제 dataset.py가 processed 폴더에서 세 데이터셋을 독립적으로 읽을 수 있다.
    print("데이터 분리 완료")


if __name__ == "__main__":
    main()
