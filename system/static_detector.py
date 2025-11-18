# static_detector.py

import os
import cv2
import numpy as np
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import slic


# ------------------------------------------------------
# util: YOLO 예측용 ROI 단위 확률 반환 함수
# ------------------------------------------------------
def make_predict_fn_for_roi(model: YOLO, class_id: int, imgsz: int = 224):
    def predict(batch_rgb):
        batch_bgr = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in batch_rgb]
        try:
            results = model.predict(source=batch_bgr, imgsz=imgsz, verbose=False)
        except:
            return np.array([[1, 0]] * len(batch_rgb), dtype=np.float32)

        probs = []
        for res in results:
            score = 0.0
            if getattr(res, "boxes", None) is not None:
                for b in res.boxes:
                    if b.conf is None or b.cls is None:
                        continue
                    if int(b.cls[0]) == class_id:
                        score = max(score, float(b.conf[0]))
            probs.append([1 - score, score])
        return np.array(probs, dtype=np.float32)

    return predict


# ------------------------------------------------------
# util: LIME 마스크 생성 (단일 ROI)
# ------------------------------------------------------
def lime_on_roi(roi_bgr, model, class_id, samples=40, n_seg=40):
    H, W = roi_bgr.shape[:2]

    # LIME 비용 최적화를 위한 축소
    small_w = max(64, min(160, W // 2))
    small_h = max(64, min(160, H // 2))
    small = cv2.resize(roi_bgr, (small_w, small_h))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    explainer = lime_image.LimeImageExplainer()

    def segmenter(img):
        return slic(img, n_segments=n_seg, compactness=8, sigma=1, start_label=0)

    predict_fn = make_predict_fn_for_roi(model, class_id, imgsz=max(small_w, small_h))

    try:
        explanation = explainer.explain_instance(
            small_rgb,
            classifier_fn=predict_fn,
            top_labels=[1],
            hide_color=0,
            num_samples=samples,
            segmentation_fn=segmenter,
        )
    except:
        return np.zeros((H, W), np.float32)

    label = explanation.top_labels[0]
    seg = explanation.segments
    exp = explanation.local_exp[label]

    pos_mask_small = np.zeros((small_h, small_w), np.float32)

    # 상위 3개만 반영
    exp = sorted(exp, key=lambda x: abs(x[1]), reverse=True)[:3]
    max_w = max(abs(w) for _, w in exp) if exp else 1e-6

    for seg_id, weight in exp:
        if weight <= 0:
            continue
        pos_mask_small[seg == seg_id] = max(pos_mask_small.max(), weight)

    pos_mask_small = np.clip(pos_mask_small / max_w, 0, 1)

    # 원래 ROI 크기로 보간
    pos_mask = cv2.resize(pos_mask_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return pos_mask


# ------------------------------------------------------
# util: 마스크 덧씌우기
# ------------------------------------------------------
def blend_mask(frame, mask, alpha=0.6):
    if mask is None or mask.max() == 0:
        return frame

    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(mask, (0, 0), 2.0)
    m3 = cv2.merge([blur, blur, blur])

    red = np.zeros_like(frame)
    red[:] = (0, 0, 255)

    out = frame.astype(np.float32) * (1 - alpha * m3) + red.astype(np.float32) * (
        alpha * m3
    )
    return out.astype(np.uint8)


# ------------------------------------------------------
# ⭐ 최종: 정적 이미지 전용 Detector
# ------------------------------------------------------
class StaticCollisionDetector:
    def __init__(self, weights_path="models/best.pt"):
        self.yolo = YOLO(weights_path)
        self.names = getattr(self.yolo.model, "names", None)

        self.CONF_THRES = 0.35
        self.MIN_CONF_LIME = 0.5
        self.LIME_SAMPLES = 40

    # --------------------------------------------------
    # 이미지 한 장 분석
    # --------------------------------------------------
    def analyze_image(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # YOLO 예측
        results = self.yolo.predict(
            frame_bgr, imgsz=320, conf=self.CONF_THRES, verbose=False
        )

        if not results or getattr(results[0], "boxes", None) is None:
            return {
                "boxes": [],
                "lime_mask": np.zeros((h, w), np.float32),
                "output": frame_bgr,
            }

        boxes = []
        for b in results[0].boxes:
            if b.conf is None or b.cls is None:
                continue
            conf = float(b.conf[0])
            if conf < self.CONF_THRES:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            cls = int(b.cls[0])
            boxes.append((x1, y1, x2, y2, cls, conf))

        # 위험도 가장 높은 박스 선택
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
        lime_mask_full = np.zeros((h, w), np.float32)

        if boxes:
            x1, y1, x2, y2, cls, conf = boxes[0]

            if conf >= self.MIN_CONF_LIME:
                roi = frame_bgr[y1:y2, x1:x2]
                pos_mask = lime_on_roi(roi, self.yolo, cls, samples=self.LIME_SAMPLES)
                # 전체 프레임에 반영
                lime_mask_full[y1:y2, x1:x2] = pos_mask

        # 최종 이미지
        blended = blend_mask(frame_bgr.copy(), lime_mask_full)

        # 박스 시각화
        for x1, y1, x2, y2, cls, conf in boxes:
            color = (0, int(255 * (1 - conf)), int(255 * conf))
            label = f"{self.names[cls]} {conf:.2f}"
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                blended,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return {
            "boxes": boxes,
            "lime_mask": lime_mask_full,
            "output": blended,
        }
