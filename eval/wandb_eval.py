# wandb 기반 모델 성능 평가 파일
import os
import numpy as np
import wandb
import weave
from dotenv import load_dotenv
from ultralytics import YOLO
from openai import OpenAI
import multiprocessing

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

os.environ["WANDB_API_KEY"] = WANDB_API_KEY
weave.init("tlswldk122104-gnu/intro-example")

# 객체 탐지 모델 평가
def evaluate_yolo():
    model_path = "YOLO-Continued/train9_finetune/weights/best.pt"
    data_yaml = "data.yaml"

    print(f"\n📊 평가 시작: {model_path}")
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

    # 클래스별 지표
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
