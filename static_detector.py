import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import slic


# =================== 유틸 함수 ===================
def _blend_single(bg: np.ndarray, fg_color_bgr: tuple, mask: np.ndarray, alpha=0.65):
    if mask is None or np.max(mask) == 0:
        return bg
    m = cv2.GaussianBlur(mask, (0, 0), 2.5)
    m3 = cv2.merge([m, m, m])
    fg = np.zeros_like(bg)
    fg[:] = fg_color_bgr
    out = bg.astype(np.float32) * (1.0 - alpha * m3) + fg.astype(np.float32) * (
        alpha * m3
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def lime_mask_on_roi_weighted(
    roi_bgr: np.ndarray, model: YOLO, class_id: int, num_samples=100
):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    h, w = roi_bgr.shape[:2]

    def segmenter(img):
        return slic(img, n_segments=70, compactness=10.0, sigma=1, start_label=0)

    explainer = lime_image.LimeImageExplainer()

    def predict_fn(batch_rgb: np.ndarray):
        bgr_batch = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in batch_rgb]
        try:
            results = model.predict(source=bgr_batch, verbose=False, imgsz=320)
        except:
            return np.array([[1.0, 0.0]] * len(batch_rgb), dtype=np.float32)
        probs = []
        for res in results:
            score = 0.0
            if getattr(res, "boxes", None):
                for bx in res.boxes:
                    if bx.conf is None or bx.cls is None:
                        continue
                    if int(bx.cls[0]) == class_id:
                        score = max(score, float(bx.conf[0]))
            pos = float(np.clip(score, 0.0, 1.0))
            probs.append([1.0 - pos, pos])
        return np.array(probs, dtype=np.float32)

    try:
        explanation = explainer.explain_instance(
            roi_rgb,
            classifier_fn=predict_fn,
            top_labels=[1],
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmenter,
        )
        label = explanation.top_labels[0]
        segments = explanation.segments
        local_exp = explanation.local_exp[label]
        # 상위 3개 슈퍼픽셀만
        sorted_exp = sorted(local_exp, key=lambda item: abs(item[1]), reverse=True)[:3]
        pos_mask = np.zeros((h, w), np.float32)
        for seg_id, weight in sorted_exp:
            mask_area = segments == seg_id
            if weight > 0:
                pos_mask[mask_area] = weight
        max_weight = max(abs(w) for _, w in sorted_exp) if sorted_exp else 1.0
        pos_mask = np.clip(pos_mask / max_weight, 0, 1)
        return pos_mask, None
    except:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)


# =================== CollisionDetectorImage ===================
class CollisionDetectorImage:
    def __init__(
        self,
        yolo_weights="models/best.pt",
        collision_model_path="models/model_weights.h5",
    ):
        self.yolo = YOLO(yolo_weights)
        self.names = getattr(self.yolo.model, "names", None)
        try:
            self.collision_model = load_model(collision_model_path)
        except:
            self.collision_model = None
        self.CLASS_HEIGHTS = {0: 1.5, 1: 5.0, 2: 10.0, 3: 1.7, 4: 0.5}
        self.FOCAL_LENGTH_PIXELS = 1400

    def _estimate_distance(self, box):
        y1, y2, cls = box[1], box[3], box[4]
        h_pixels = max(y2 - y1, 1)
        H_actual = self.CLASS_HEIGHTS.get(cls, 1.7)
        return (H_actual * self.FOCAL_LENGTH_PIXELS) / h_pixels

    def _draw_boxes(self, frame, boxes, conf_thres=0.35):
        infos = []
        for b in boxes:
            conf = b[5]
            if conf < conf_thres:
                continue
            x1, y1, x2, y2, cls = b[:5]
            color = (0, int(255 * (1 - conf)), int(255 * conf))
            label = self.names[cls] if self.names else str(cls)
            label = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            infos.append((x1, y1, x2, y2, cls, conf))
        return frame, infos

    def process_image(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, {"error": "이미지 읽기 실패"}, None

        results = self.yolo.predict(frame, imgsz=320, verbose=False)
        processed_frame, boxes = self._draw_boxes(
            frame.copy(),
            [
                (
                    int(b.xyxy[0][0]),
                    int(b.xyxy[0][1]),
                    int(b.xyxy[0][2]),
                    int(b.xyxy[0][3]),
                    int(b.cls[0]),
                    float(b.conf[0]),
                )
                for b in results[0].boxes
            ],
        )

        # 충돌 분류
        collision_prob = 0.0
        closest_box = None
        min_dist = float("inf")
        for b in boxes:
            dist = self._estimate_distance(b)
            if dist < min_dist:
                min_dist = dist
                closest_box = b

        if closest_box and self.collision_model:
            x1, y1, x2, y2, cls, conf = closest_box
            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                roi_input = cv2.resize(roi, (128, 128)).astype(np.float32) / 255.0
                roi_input = roi_input[np.newaxis, ...]
                pred = self.collision_model.predict(roi_input, verbose=0)[0]
                collision_prob = float(pred[1]) if len(pred) > 1 else float(pred[0])

        max_conf = (
            collision_prob if self.collision_model else (boxes[0][5] if boxes else 0.0)
        )

        # LIME 마스크
        mask_pos = None
        if max_conf >= 0.5 and closest_box:
            x1, y1, x2, y2, cls, conf = closest_box
            roi = frame[y1:y2, x1:x2]
            mask_pos, _ = lime_mask_on_roi_weighted(roi, self.yolo, cls)
            mask_full = np.zeros(frame.shape[:2], np.float32)
            mask_full[y1:y2, x1:x2] = mask_pos
            mask_pos = mask_full
            frame = _blend_single(frame, (0, 0, 255), mask_pos, alpha=0.65)

        return processed_frame, {"max_conf": max_conf}, mask_pos
