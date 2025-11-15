from ultralytics import YOLO
import torch
import numpy as np
import os

def get_class_weights():
    class_counts = {
        "car": 189504,
        "tree": 395,
        "person": 106962,
        "other": 49002
    }

    total = sum(class_counts.values())
    weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

    print("클래스별 가중치 (비교용):")
    for k, v in weights.items():
        print(f" - {k:7s}: {v:.6f}")

    return weights

def train_and_evaluate():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 기존 모델 로드
    model_path = "YOLO-Continued\\train_balance_finetune_v3\\weights\\best.pt"
    model = YOLO(model_path)

    # Fine-tuning 학습
    results = model.train(
        data="data.yaml",
        epochs=100,                 # 더 길게 학습
        batch=16,
        imgsz=640,
        lr0=0.0003,                  # 더 낮은 lr로 정밀 조정
        lrf=0.01,                   # cosine 최종 lr
        optimizer="AdamW",
        device=device,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,                # Cosine LR 스케줄러
        freeze=0,                  # Backbone 일부 고정

        # Data Augmentation 강화
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5, translate=0.05, scale=0.5, shear=1.0,
        flipud=0.2, fliplr=0.5,
        mosaic=0.4, mixup=0.1, copy_paste=0.3,

        # Early stopping
        patience=15,

        # 저장 경로
        project="YOLO-Continued",
        name="train_balance_finetune_v3",
        exist_ok=True,
        verbose=True
    )

    # 학습 후 평가
    metrics = model.val(data="data.yaml", device=device, split='val', plots=True)

    # F1 계산
    f1 = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    # 결과 출력
    print("\n=== 평가 결과 ===")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision mean: {np.mean(metrics.box.p):.4f}")
    print(f"Recall mean   : {np.mean(metrics.box.r):.4f}")
    print(f"F1 mean       : {np.mean(f1):.4f}")

    # 결과 저장
    save_dir = os.path.join("YOLO-Continued", "train_balance_finetune_v3", "results_summary.txt")
    with open(save_dir, "w") as f:
        f.write("=== YOLOv8 Fine-tune Evaluation ===\n")
        f.write(f"mAP@0.5      : {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95 : {metrics.box.map:.4f}\n")
        f.write(f"Precision mean: {np.mean(metrics.box.p):.4f}\n")
        f.write(f"Recall mean   : {np.mean(metrics.box.r):.4f}\n")
        f.write(f"F1 mean       : {np.mean(f1):.4f}\n")

    print(f"평가 요약 저장됨: {save_dir}")

if __name__ == "__main__":
    get_class_weights()
    train_and_evaluate()
