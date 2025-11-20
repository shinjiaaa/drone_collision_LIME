from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from plyer import notification

app = FastAPI()

# === YOLO 모델 로드 ===
yolo_model = YOLO("models/best.pt")

# === Keras 충돌 분류 모델 로드 ===
collision_model = load_model("models/model_weights.h5")

# === 모델이 기대하는 입력 크기 확인 ===
input_height, input_width = collision_model.input_shape[1:3]


# === 이미지 전처리 함수 ===
def preprocess_for_collision_model(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (input_width, input_height))  # 모델 입력 크기에 맞춤
    crop = crop.astype(np.float32) / 255.0
    crop = np.expand_dims(crop, axis=0)
    return crop


# === LIME 예측 함수 ===
def predict_for_lime(images):
    results = []
    for img in images:
        img_input = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
        prob = collision_model.predict(img_input, verbose=0)[0][0]
        results.append([prob, 1 - prob])
    return np.array(results)


# === 업로드 페이지 ===
@app.get("/")
async def upload_page():
    return HTMLResponse(
        """
    <html>
        <body>
            <h2>Drone Collision Analysis</h2>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
    )


# === 업로드 후 예측 ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 이미지 읽기
    file_bytes = await file.read()
    pil_image = Image.open(BytesIO(file_bytes)).convert("RGB")
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # YOLO 객체 감지
    yolo_results = yolo_model.predict(img_rgb, verbose=False)
    bboxes = []
    for r in yolo_results:
        for box in r.boxes.xyxy:
            bboxes.append(box.cpu().numpy())

    # 충돌 분류
    conf_threshold = 0.7
    collision_conf = 0.0
    for bbox in bboxes:
        obj_input = preprocess_for_collision_model(img_rgb, bbox)
        prob = collision_model.predict(obj_input, verbose=0)[0][0]
        collision_conf = max(collision_conf, prob)

    # 충돌 알람 & LIME 적용
    if collision_conf >= conf_threshold:
        notification.notify(
            title="Drone Collision Warning!",
            message=f"Collision Risk Detected!\nProbability: {collision_conf*100:.1f}%",
            timeout=10,
        )

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_rgb,
            classifier_fn=predict_for_lime,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
        )

        label = explanation.top_labels[0]
        positive_sum = sum(
            weight for _, weight in explanation.local_exp[label] if weight > 0
        )
        negative_sum = sum(
            weight for _, weight in explanation.local_exp[label] if weight < 0
        )
        total = positive_sum + abs(negative_sum)

        print(f"\n=== 충돌 위험 분석 ===")
        print(f"충돌 확률: {collision_conf*100:.1f}%")
        print(f"충돌 위험 영역 기여도: {positive_sum/total*100:.2f}%")
        print(f"안전 영역 기여도: {abs(negative_sum)/total*100:.2f}%")

        # === 시각화 ===
        temp_pos, mask_pos = explanation.get_image_and_mask(
            label, positive_only=True, num_features=2, hide_rest=False
        )
        temp_neg, mask_neg = explanation.get_image_and_mask(
            label,
            positive_only=False,
            negative_only=True,
            num_features=2,
            hide_rest=False,
        )

        base_img = img_rgb / 255.0
        pos_overlay = base_img.copy()
        pos_overlay[mask_pos] = [1, 0, 0]
        neg_overlay = base_img.copy()
        neg_overlay[mask_neg] = [0, 1, 0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(base_img)
        ax.imshow(pos_overlay, alpha=0.4)
        ax.imshow(neg_overlay, alpha=0.4)
        ax.imshow(
            mark_boundaries(base_img, mask_pos, color=(1, 0, 0), mode="thick"),
            alpha=0.8,
        )
        ax.imshow(
            mark_boundaries(base_img, mask_neg, color=(0, 1, 0), mode="thick"),
            alpha=0.8,
        )
        ax.axis("off")
        plt.show()
    else:
        print(f"[INFO] Low collision risk ({collision_conf*100:.1f}%)")

    return {"collision_probability": collision_conf}
