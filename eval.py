"""
YOLO 모델 평가 전용 스크립트 (AI 리포트 X)
이미 학습된 모델(best.pt)을 불러와 성능 평가 + W&B 로그 기록
"""
import os
import numpy as np
import wandb
from dotenv import load_dotenv
from ultralytics import YOLO
import multiprocessing

# 1️⃣ 환경 변수 로드 (.env 파일에서 API 키 불러오기)
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def evaluate_yolo():
    model_path = "runs/detect/VisDrone_train4/weights/best.pt"
    data_yaml = "VisDrone.yaml"

    model = YOLO(model_path)

    # 모델 평가
    metrics = model.val(data=data_yaml, device=0)
    f1_score = 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6)

    print("\n=== 평가 결과 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {np.mean(metrics.box.p):.4f}")
    print(f"Recall    (mean): {np.mean(metrics.box.r):.4f}")
    print(f"F1-score (mean): {np.mean(f1_score):.4f}")

    # 클래스별 상세 지표
    print("\n=== 클래스별 Precision / Recall ===")
    print("Precision:", np.round(metrics.box.p, 4))
    print("Recall   :", np.round(metrics.box.r, 4))

    # W&B 로깅
    wandb.init(project="YOLO-Evaluation", name="VisDrone_best_eval")
    wandb.log({
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision_mean": np.mean(metrics.box.p),
        "Recall_mean": np.mean(metrics.box.r),
        "F1_score_mean": np.mean(f1_score)
    })
    wandb.finish()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate_yolo()
