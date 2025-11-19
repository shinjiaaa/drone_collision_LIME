import os, time, cv2, json
import numpy as np
from typing import Optional, Tuple, Dict, Any
from threading import Thread, Lock, Event
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import slic
from tensorflow.keras.models import load_model
from explainer import generate_lime_explanation

# 카메라 캘리브레이션 설정
CLASS_HEIGHTS = {0: 1.5, 1: 5.0, 2: 10.0, 3: 1.7, 4: 0.5}
FOCAL_LENGTH_PIXELS = 1400

# 모델 경로
DEFAULT_YOLO = "yolo11n.pt"  # 객체 탐지 모델
DEFAULT_COLLISION = "models/model_weights.h5"  # 충돌 분류 모델

# --------------------------
# 유틸 함수
# --------------------------
def estimate_distance(box, class_id):
    y1, y2 = box[1], box[3]
    h_pixels = max(y2 - y1, 1)
    H_actual = CLASS_HEIGHTS.get(class_id, 1.7)
    return (H_actual * FOCAL_LENGTH_PIXELS) / h_pixels


def draw_risk_indicator(frame, max_conf, warning_threshold):
    h, w = frame.shape[:2]
    bar_h = 20
    cv2.rectangle(frame, (0, 0), (w, bar_h), (50, 50, 50), -1)
    risk_w = int(w * max_conf)
    if max_conf < 0.5:
        r, g = int(255 * (max_conf * 2)), 255
    else:
        r, g = 255, int(255 * (1 - (max_conf - 0.5) * 2))
    color = (0, g, r)
    if risk_w > 0:
        cv2.rectangle(frame, (0, 0), (risk_w, bar_h), color, -1)
    tx = int(w * warning_threshold)
    if 0 <= tx < w:
        cv2.line(frame, (tx, 0), (tx, bar_h), (255, 255, 255), 2)
    text = f"RISK LEVEL: {max_conf*100:.1f}%"
    tc = (255, 255, 255) if max_conf < 0.6 else (10, 10, 10)
    cv2.putText(
        frame, text, (10, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1, cv2.LINE_AA
    )


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
        (w_text, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, h + 30)
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        boxes_info.append((x1, y1, x2, y2, cls, conf))
    return frame, boxes_info


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


def blend_dual_mask_sequential(
    frame_bgr: np.ndarray,
    pos_mask01: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if pos_mask01 is None or pos_mask01.shape != (h, w):
        return frame_bgr
    COLOR_RED = (0, 0, 255)
    return _blend_single(frame_bgr, COLOR_RED, pos_mask01, alpha)


# --------------------------
# LIME
# --------------------------
def make_predict_fn_for_roi(model: YOLO, class_id: int):
    def predict_proba(batch_rgb: np.ndarray) -> np.ndarray:
        bgr_batch = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in batch_rgb]
        try:
            results = model.predict(source=bgr_batch, verbose=False, imgsz=320)
        except Exception as e:
            print(f"[WARN] YOLO batch prediction error: {e}")
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
            pos = float(np.clip(score, 0.0, 1.0))
            probs.append([1.0 - pos, pos])
        return np.array(probs, dtype=np.float32)
    return predict_proba


def lime_mask_on_roi_weighted(
    roi_bgr: np.ndarray,
    model: YOLO,
    class_id: int,
    num_samples: int,
    n_segments=70,
    compactness=10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    h, w = roi_bgr.shape[:2]

    def segmenter(img):
        return slic(
            img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=0
        )

    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_predict_fn_for_roi(model, class_id)
    try:
        explanation = explainer.explain_instance(
            roi_rgb,
            classifier_fn=predict_fn,
            top_labels=[1],
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmenter,
        )

        if not hasattr(explanation, "top_labels") or len(explanation.top_labels) == 0:
            print("[LIME] no top_labels in explanation")
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        label = explanation.top_labels[0]
        segments = explanation.segments
        local_exp = explanation.local_exp if hasattr(explanation, "local_exp") else {}

        if label not in local_exp:
            print(f"[LIME] label {label} not in local_exp keys")
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        # 상위 3개 슈퍼픽셀
        sorted_exp = sorted(local_exp[label], key=lambda item: abs(item[1]), reverse=True)[:3]
        pos_mask = np.zeros((h, w), dtype=np.float32)
        neg_mask = np.zeros((h, w), dtype=np.float32)

        for seg_id, weight in sorted_exp:
            mask_area = segments == seg_id
            if weight > 0:
                pos_mask[mask_area] = weight
            elif weight < 0:
                neg_mask[mask_area] = abs(weight)

        max_weight = max(abs(w) for _, w in sorted_exp) if sorted_exp else 1.0
        pos_mask = np.clip(pos_mask / max_weight, 0.0, 1.0)
        neg_mask = np.clip(neg_mask / max_weight, 0.0, 1.0)
        return pos_mask, neg_mask
    except Exception as e:
        print(f"[LIME ERROR] {e}")
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)


# --------------------------
# CollisionDetectorLIME
# --------------------------
class CollisionDetectorLIME:
    def __init__(self, weights_path: Optional[str] = None, collision_model_path: Optional[str] = None):
        self.config = {
            "imgsz": 320,
            "conf_thres": 0.35,
            "min_conf_for_lime": 0.5,
            "warning_threshold": 0.75,
            "roi_shrink": 96,
            "topk": 1,
            "lime_samples": 100,
            "lime_alpha": 0.65,
        }
        self.config_lock = Lock()

        if weights_path and os.path.exists(weights_path):
            self.weights = weights_path
        elif os.path.exists(DEFAULT_YOLO):
            self.weights = DEFAULT_YOLO
        else:
            self.weights = self._find_weights(weights_path)
        print(f"[Detector] Loading YOLO model from: {self.weights}")
        self.yolo = YOLO(self.weights)
        self.names = getattr(self.yolo.model, "names", None)

        self.collision_model = None
        if collision_model_path and os.path.exists(collision_model_path):
            try:
                self.collision_model = load_model(collision_model_path)
            except Exception as e:
                print(f"[Detector] Failed to load collision model: {e}")
        elif os.path.exists(DEFAULT_COLLISION):
            try:
                self.collision_model = load_model(DEFAULT_COLLISION)
            except Exception as e:
                print(f"[Detector] Failed to load default collision model: {e}")

        self.last_mask_pos: Optional[np.ndarray] = None
        self.last_mask_neg: Optional[np.ndarray] = None
        self.last_lime_json_time: float = 0.0
        self.frame_count: int = 0
        self.latest_lime_mask_img = None


        self.data_lock = Lock()
        self.cancel_event = Event()
        self.latest_job = {"frame": None, "boxes": None}
        self.worker_thread: Optional[Thread] = None

        self.t0, self.cnt, self.fps = time.time(), 0, 0.0
        self.last_alert_time = 0

    def _find_weights(self, path):
        candidates = ["best.pt", "yolo11n.pt"]
        return next((c for c in candidates if os.path.exists(c)), candidates[-1])

    def get_config(self) -> Dict[str, Any]:
        with self.config_lock:
            return self.config.copy()

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.cancel_event.clear()
            self.worker_thread = Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        self.cancel_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=3.0)

    # --------------------------
    # 백그라운드 LIME
    # --------------------------
    def _worker_loop(self):
        frame_count = 0
        while not self.cancel_event.is_set():
            job_frame, job_boxes = None, None
            with self.data_lock:
                if self.latest_job["frame"] is not None and self.latest_job["boxes"]:
                    job_frame = self.latest_job["frame"]
                    job_boxes = self.latest_job["boxes"]
                    self.latest_job["frame"] = None
                    self.latest_job["boxes"] = None

            if job_frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            N = 5  # 매 5프레임마다 LIME 계산
            if frame_count % N != 0:
                time.sleep(0.01)
                continue

            cfg = self.get_config()
            H, W = job_frame.shape[:2]

            sel = job_boxes[: cfg["topk"]]
            if not sel:
                time.sleep(0.02)
                continue

            mask_full_pos = np.zeros((H, W), np.float32)

            for x1, y1, x2, y2, cls, conf in sel:
                if self.cancel_event.is_set():
                    return

                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = job_frame[y1:y2, x1:x2]

                # --- ROI 최소 크기 확보 ---
                min_size = 64
                roi_h, roi_w = roi.shape[:2]
                if roi_h < min_size or roi_w < min_size:
                    roi_resized = cv2.resize(roi, (max(roi_w, min_size), max(roi_h, min_size)))
                else:
                    roi_resized = roi.copy()

                # --- LIME 마스크 생성 ---
                pos_mask, _ = lime_mask_on_roi_weighted(
                    roi_resized, self.yolo, cls, num_samples=cfg["lime_samples"]
                )

                # --- 원래 ROI 크기로 리사이즈 ---
                pos_mask_resized = cv2.resize(pos_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

                # --- 디버깅용 출력 ---
                print(f"[DEBUG] ROI size: {roi.shape}, pos_mask max: {pos_mask.max()}")

                # --- 전체 프레임에 마스크 적용 준비 ---
                full_mask = np.zeros((H, W), np.float32)
                full_mask[y1:y2, x1:x2] = pos_mask_resized
                mask_full_pos = np.maximum(mask_full_pos, full_mask)

            with self.data_lock:
                self.last_mask_pos = mask_full_pos
                self.last_mask_neg = np.zeros_like(mask_full_pos)

                # --- LIME 마스크 저장 (프론트에서 가져갈 이미지) ---
                # 0~1 → 0~255 흰색 마스크로 변환
                lime_img = (mask_full_pos * 255).astype(np.uint8)

                # 3채널 BGR 변환 (프론트 전송용)
                self.latest_lime_mask_img = cv2.cvtColor(lime_img, cv2.COLOR_GRAY2BGR)



    # --------------------------
    # 위험도 평가
    # --------------------------
    def _evaluate_risk(self, max_conf: float) -> Dict[str, Any]:
        alert_event = None
        if max_conf >= 0.80:
            level, text = "danger", "위험"
            sound, tts = "alert_high_repeat", "충돌 위험"
        elif max_conf >= 0.60:
            level, text = "warning", "경고"
            sound, tts = "alert_mid_2", "경고"
        elif max_conf >= 0.50:
            level, text = "caution", "주의"
            sound, tts = "alert_low_1", "주의"
        else:
            level, text = "safe", "안전"
            sound, tts = None, None

        now = time.time()
        is_risky = level != "safe"
        if is_risky and now - self.last_alert_time > 2.0:
            self.last_alert_time = now
            alert_event = {
                "level": level,
                "message": f"충돌 위험 감지: {max_conf*100:.1f}%",
                "sound": sound,
                "tts": tts,
            }
        return {
            "max_conf": max_conf,
            "level": level,
            "text": text,
            "alert_event": alert_event,
        }

    # --------------------------
    # 메인 처리
    # --------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame_bgr is None:
            return frame_bgr, self._evaluate_risk(0.0)

        cfg = self.get_config()
        results = self.yolo.predict(source=frame_bgr, imgsz=cfg["imgsz"], verbose=False)
        processed_frame, boxes = draw_boxes(frame_bgr.copy(), results, cfg["conf_thres"], self.names)

        # 가장 가까운 객체 선택
        min_distance = float("inf")
        closest_box = None
        for box in boxes:
            x1, y1, x2, y2, cls, conf = box
            distance = estimate_distance(box, cls)
            if distance < min_distance:
                min_distance = distance
                closest_box = box

        collision_prob = 0.0
        if closest_box is not None and self.collision_model is not None:
            x1, y1, x2, y2, cls, conf = closest_box
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size != 0:
                try:
                    roi_resized = cv2.resize(roi, (128, 128))
                    roi_input = (roi_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
                    pred = self.collision_model.predict(roi_input, verbose=0)[0]
                    collision_prob = float(pred[1]) if len(pred) >= 2 else float(pred[0])
                    distance_factor = np.exp(-min_distance / 10.0)
                    collision_prob *= 0.5 + 0.5 * distance_factor
                    collision_prob = np.clip(collision_prob, 0.0, 1.0)
                except Exception as e:
                    print(f"[Detector WARN] collision model predict error: {e}")
                    collision_prob = 0.0

        max_conf = collision_prob if self.collision_model else (boxes[0][5] if boxes else 0.0)

        # LIME 작업 전달
        if boxes and max_conf >= cfg["min_conf_for_lime"]:
            with self.data_lock:
                self.latest_job["frame"] = frame_bgr.copy()
                self.latest_job["boxes"] = boxes
        else:
            with self.data_lock:
                self.latest_job["frame"] = None
                self.latest_job["boxes"] = None

        # 마스크 적용
        mask_pos = self.last_mask_pos
        if mask_pos is not None:
            processed_frame = blend_dual_mask_sequential(
                processed_frame, mask_pos, alpha=cfg["lime_alpha"]
            )

        # 위험도 표시
        draw_risk_indicator(processed_frame, max_conf, cfg["warning_threshold"])
        risk_info = self._evaluate_risk(max_conf)

        return processed_frame, risk_info

    def _calculate_fps(self):
        self.cnt += 1
        now = time.time()
        if now - self.t0 >= 0.5:
            self.fps = self.cnt / (now - self.t0)
            self.t0, self.cnt = now, 0
