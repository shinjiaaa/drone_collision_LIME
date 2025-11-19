# system/static_detector.py
import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import slic
from tensorflow.keras.models import load_model

# -----------------------
# 기본 경로 & 카메라 파라미터 (필요시 조정)
# -----------------------
DEFAULT_YOLO = "models/best.pt"
DEFAULT_COLLISION = "models/model_weights.h5"
FOCAL_LENGTH_PIXELS = 1400
CLASS_HEIGHTS = {0: 1.5, 1: 5.0, 2: 10.0, 3: 1.7, 4: 0.5}


# -----------------------
# 유틸: 거리 추정 (optional, collision_prob 보정용)
# -----------------------
def estimate_distance(box, class_id):
    # box: (x1,y1,x2,y2,cls,conf)
    y1, y2 = box[1], box[3]
    h_pixels = max(y2 - y1, 1)
    H_actual = CLASS_HEIGHTS.get(class_id, 1.7)
    return (H_actual * FOCAL_LENGTH_PIXELS) / h_pixels


# -----------------------
# 유틸: YOLO 결과로 박스 그리기 및 box list 반환
# -----------------------
def draw_boxes(frame, results, conf_thres=0.35, names=None):
    if not results or getattr(results[0], "boxes", None) is None:
        return frame, []
    sorted_boxes = sorted(
        results[0].boxes,
        key=lambda b: float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0,
        reverse=True,
    )
    boxes_info = []
    for b in sorted_boxes:
        if b.conf is None or b.xyxy is None:
            continue
        conf = float(b.conf[0])
        if conf < conf_thres:
            break
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        cls = int(b.cls[0]) if b.cls is not None else -1
        color = (0, int(255 * (1 - conf)), int(255 * conf))
        label = names[cls] if names and 0 <= cls < len(names) else str(cls)
        label = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, h_text + 20)
        cv2.putText(
            frame,
            label,
            (x1, text_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        boxes_info.append((x1, y1, x2, y2, cls, conf))
    return frame, boxes_info


# -----------------------
# 유틸: 마스크 블렌드 (red for positive)
# -----------------------
def _blend_single(
    bg: np.ndarray, fg_color_bgr: Tuple[int, int, int], mask: np.ndarray, alpha: float
) -> np.ndarray:
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


def blend_pos_mask(
    frame_bgr: np.ndarray, pos_mask01: np.ndarray, alpha: float = 0.6
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if pos_mask01 is None or pos_mask01.shape != (h, w):
        return frame_bgr
    COLOR_RED = (0, 0, 255)
    return _blend_single(frame_bgr, COLOR_RED, pos_mask01, alpha)


# -----------------------
# LIME: ROI 단위 predict wrapper (YOLO 기반 또는 collision classifier 기반)
# -----------------------
def make_predict_fn_for_roi_yolo(model: YOLO, class_id: int, imgsz: int = 160):
    def predict(batch_rgb):
        bgr_batch = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in batch_rgb]
        try:
            results = model.predict(source=bgr_batch, imgsz=imgsz, verbose=False)
        except Exception:
            return np.array([[1.0, 0.0]] * len(batch_rgb), dtype=np.float32)
        probs = []
        for res in results:
            score = 0.0
            if getattr(res, "boxes", None) is not None:
                for bx in res.boxes:
                    if bx.conf is None or bx.cls is None:
                        continue
                    if int(bx.cls[0]) == class_id:
                        score = max(score, float(bx.conf[0]))
            probs.append([1.0 - float(score), float(score)])
        return np.array(probs, dtype=np.float32)

    return predict


# -----------------------
# LIME 마스크 생성: 상위 3개 슈퍼픽셀 (원본 코드와 호환)
# -----------------------
def lime_mask_on_roi_weighted(
    roi_bgr: np.ndarray,
    model: YOLO,
    class_id: int,
    num_samples: int = 100,
    n_segments=70,
    compactness=10.0,
):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    h, w = roi_bgr.shape[:2]

    def segmenter(img):
        return slic(
            img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=0
        )

    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_predict_fn_for_roi_yolo(model, class_id, imgsz=max(64, min(w, h)))
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
        sorted_exp = sorted(local_exp, key=lambda item: abs(item[1]), reverse=True)[:3]
        pos_mask = np.zeros((h, w), dtype=np.float32)
        neg_mask = np.zeros((h, w), dtype=np.float32)
        if not sorted_exp:
            return pos_mask, neg_mask
        for seg_id, weight in sorted_exp:
            mask_area = segments == seg_id
            if weight > 0:
                pos_mask[mask_area] = weight
            elif weight < 0:
                neg_mask[mask_area] = abs(weight)
        max_weight = max(abs(wt) for _, wt in sorted_exp)
        if max_weight > 0:
            pos_mask = np.clip(pos_mask / max_weight, 0.0, 1.0)
            neg_mask = np.clip(neg_mask / max_weight, 0.0, 1.0)
        return pos_mask, neg_mask
    except Exception:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)


# -----------------------
# 최종: StaticCollisionDetectorLIME (정적 이미지 전용)
# -----------------------
class StaticCollisionDetectorLIME:
    def __init__(
        self,
        yolo_weights: str = DEFAULT_YOLO,
        collision_weights: str = DEFAULT_COLLISION,
    ):
        # YOLO 로드
        self.yolo = (
            YOLO(yolo_weights) if os.path.exists(yolo_weights) else YOLO(DEFAULT_YOLO)
        )
        self.names = getattr(self.yolo.model, "names", None)

        # Collision classifier 로드(옵션)
        self.collision_model = None
        if collision_weights and os.path.exists(collision_weights):
            try:
                self.collision_model = load_model(collision_weights)
            except Exception as e:
                print(f"[StaticDetector] failed to load collision model: {e}")
                self.collision_model = None

        # 기본 설정
        self.imgsz = 320
        self.conf_thres = 0.35
        self.min_conf_for_lime = 0.5
        self.topk = 1
        self.lime_samples = 100
        self.lime_n_segments = 70
        self.lime_compactness = 10.0
        self.lime_alpha = 0.65

    def _predict_collision_from_roi(self, roi_bgr: np.ndarray) -> float:
        if self.collision_model is None:
            return 0.0
        try:
            roi_resized = cv2.resize(roi_bgr, (128, 128))
            roi_input = (roi_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
            pred = self.collision_model.predict(roi_input, verbose=0)[0]
            # 모델 출력 형식에 따라 처리(softmax 또는 single prob)
            if hasattr(pred, "__len__") and len(pred) >= 2:
                prob = float(pred[1])
            else:
                prob = float(pred[0])
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as e:
            print(f"[StaticDetector] collision predict error: {e}")
            return 0.0

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        입력: BGR 이미지 (np.uint8)
        반환:
          processed_frame (BGR uint8),
          info dict: {
            "boxes": list of (x1,y1,x2,y2,cls,conf),
            "pos_mask": np.ndarray float32 (H,W) 0..1,
            "neg_mask": np.ndarray float32 (H,W) 0..1,
            "class_name": str,
            "collision_prob": float
          }
        """
        if frame_bgr is None:
            return frame_bgr, {
                "boxes": [],
                "pos_mask": None,
                "neg_mask": None,
                "class_name": None,
                "collision_prob": 0.0,
            }

        H, W = frame_bgr.shape[:2]
        results = self.yolo.predict(source=frame_bgr, imgsz=self.imgsz, verbose=False)

        processed_frame, boxes = draw_boxes(
            frame_bgr.copy(), results, conf_thres=self.conf_thres, names=self.names
        )

        # boxes: list of tuples
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
        pos_full = np.zeros((H, W), dtype=np.float32)
        neg_full = np.zeros((H, W), dtype=np.float32)
        collision_prob = 0.0
        class_name = "unknown"

        if boxes:
            # 선택(closest by estimated distance preferred; fallback to highest conf)
            closest_box = None
            min_dist = float("inf")
            for b in boxes:
                try:
                    d = estimate_distance(b, b[4])
                except Exception:
                    d = float("inf")
                if d < min_dist:
                    min_dist = d
                    closest_box = b
            sel = closest_box if closest_box is not None else boxes[0]
            x1, y1, x2, y2, cls, conf = sel
            class_name = (
                self.names[cls]
                if self.names and 0 <= cls < len(self.names)
                else str(cls)
            )

            # collision probability: classifier if available else YOLO conf
            if self.collision_model is not None:
                roi = frame_bgr[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
                if roi.size != 0:
                    collision_prob = self._predict_collision_from_roi(roi)
                    # 거리 보정 (optional)
                    try:
                        distance = estimate_distance(sel, cls)
                        if distance > 0:
                            distance_factor = np.exp(-distance / 10.0)
                            collision_prob *= 0.5 + 0.5 * distance_factor
                            collision_prob = float(np.clip(collision_prob, 0.0, 1.0))
                    except Exception:
                        pass
                else:
                    collision_prob = 0.0
            else:
                collision_prob = float(conf)

            # LIME: ROI 기반 마스크 (YOLO 기반 LIME으로 빠르게 계산)
            if collision_prob >= self.min_conf_for_lime:
                roi = frame_bgr[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
                if roi.size != 0:
                    pos_mask_roi, neg_mask_roi = lime_mask_on_roi_weighted(
                        roi,
                        self.yolo,
                        cls,
                        num_samples=self.lime_samples,
                        n_segments=self.lime_n_segments,
                        compactness=self.lime_compactness,
                    )
                    # resize if needed (should be same)
                    h_r, w_r = roi.shape[:2]
                    if pos_mask_roi.shape != (h_r, w_r):
                        pos_mask_roi = cv2.resize(
                            pos_mask_roi, (w_r, h_r), interpolation=cv2.INTER_LINEAR
                        )
                    if neg_mask_roi.shape != (h_r, w_r):
                        neg_mask_roi = cv2.resize(
                            neg_mask_roi, (w_r, h_r), interpolation=cv2.INTER_LINEAR
                        )
                    pos_full[y1:y2, x1:x2] = pos_mask_roi
                    neg_full[y1:y2, x1:x2] = neg_mask_roi

        # 시각화: pos 마스크 (빨간색) 덮기 + 박스/라벨
        vis = blend_pos_mask(processed_frame, pos_full, alpha=self.lime_alpha)
        for x1, y1, x2, y2, cls, conf in boxes:
            color = (0, int(255 * (1 - conf)), int(255 * conf))
            label = (
                self.names[cls]
                if self.names and 0 <= cls < len(self.names)
                else str(cls)
            ) + f" {conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # neg mask 시각화 (optional: blue tint with low alpha)
        if np.max(neg_full) > 0:
            neg_color = np.zeros_like(vis)
            neg_color[:] = (255, 0, 0)
            neg_3d = cv2.merge([neg_full, neg_full, neg_full])
            vis = (
                vis.astype(np.float32) * (1 - 0.25 * neg_3d)
                + neg_color.astype(np.float32) * (0.25 * neg_3d)
            ).astype(np.uint8)

        info: Dict[str, Any] = {
            "boxes": boxes,
            "pos_mask": pos_full,
            "neg_mask": neg_full,
            "class_name": class_name,
            "collision_prob": float(collision_prob),
        }
        return vis, info
