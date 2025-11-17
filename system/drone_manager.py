import time
import threading
import cv2
import numpy as np
from typing import Optional, Dict, Any, Callable
try:
    from djitellopy import Tello

    TELLO_AVAILABLE = True
except ImportError:
    Tello = None
    TELLO_AVAILABLE = False
    print("[Manager] djitellopy not found. Tello control disabled.")

try:
    from detector import CollisionDetectorLIME
except ImportError as e:
    print(f"[ERROR] detector.py를 찾을 수 없습니다. {e}")
    exit()


class DroneManager:
    def __init__(self, detector: CollisionDetectorLIME, use_webcam: bool = False):
        self.detector = detector
        self.use_webcam = use_webcam
        if not TELLO_AVAILABLE and not use_webcam:
            print(
                "[Manager] Warning: Tello requested but library missing. Forcing webcam mode."
            )
            self.use_webcam = True

        self.tello: Optional[Tello] = None
        self.webcam: Optional[cv2.VideoCapture] = None

        # 설정
        self.FRAME_WIDTH = 960
        self.FRAME_HEIGHT = 720
        self.DRONE_SPEED = 50

        # 상태 변수
        self.is_connected = False
        self.is_streaming = False
        self.battery_level = 0
        self.latest_processed_data: Dict[str, Any] = {"frame": None, "risk": None}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.latest_telemetry: Dict[str, Any] = self._generate_telemetry_snapshot(None)

    def connect(self):
        if self.is_connected:
            return True
        try:
            if self.use_webcam:
                self._connect_webcam()
            else:
                try:
                    self._connect_tello()
                except Exception as e:
                    print(
                        f"[Manager] Tello connection failed: {e}. Falling back to Webcam."
                    )
                    self.use_webcam = True
                    self._connect_webcam()
            self.is_connected = True
            self.detector.start_worker()
            self.processing_thread = threading.Thread(
                target=self._main_loop, daemon=True
            )
            self.processing_thread.start()
            print("[Manager] System started successfully.")
            return True
        except Exception as e:
            print(f"[Manager] Final connection failed: {e}")
            self.is_connected = False
            return False

    def _connect_tello(self):
        print("[Manager] Connecting to Tello...")
        self.tello = Tello()
        self.tello.connect()
        self.battery_level = self.tello.get_battery()
        if self.battery_level == 0:
            raise RuntimeError(
                "Tello connection established but communication failed (Battery 0%)."
            )
        print(f"[Manager] Tello connected. Battery: {self.battery_level}%")
        self.tello.streamon()
        self.frame_reader = self.tello.get_frame_read()
        self.tello.send_rc_control(0, 0, 0, 0)
        self.is_streaming = True

    def _connect_webcam(self):
        print("[Manager] Connecting to Webcam...")
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            raise RuntimeError("Could not open webcam.")
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        self.battery_level = 100
        self.is_streaming = True

    def shutdown(self):
        print("[Manager] Shutting down...")
        self.stop_event.set()
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        if self.detector:
            self.detector.stop_worker()
        if self.tello and not self.use_webcam:
            try:
                print("[Manager] Landing drone safely...")
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.land()
                time.sleep(3)
            except Exception as e:
                print(f"[Manager] Landing failed: {e}")
            try:
                self.tello.streamoff()
                self.tello.end()
            except:
                pass
        if self.webcam:
            self.webcam.release()
        self.is_connected = False
        print("[Manager] Shutdown complete.")

    # 메인 루프
    def _main_loop(self):
        last_battery_check = time.time()
        while not self.stop_event.is_set():
            if not self.is_streaming:
                time.sleep(0.1)
                continue

            try:
                # 1. 영상 프레임 획득
                frame_bgr = self._get_frame()
                if frame_bgr is None:
                    time.sleep(0.01)
                    continue

                # 2. AI 처리 (YOLO + LIME)
                processed_frame, risk_data = self.detector.process_frame(frame_bgr)

                # 3. 최신 데이터 저장
                with self.lock:
                    self.latest_processed_data["frame"] = processed_frame
                    self.latest_processed_data["risk"] = risk_data

                # 4. 텔레메트리 데이터 내부 상태 업데이트 & 배터리 체크
                now = time.time()
                if self.tello and not self.use_webcam and now - last_battery_check > 10:
                    try:
                        self.battery_level = self.tello.get_battery()
                        last_battery_check = now
                    except:
                        pass

                # 내부 상태 업데이트
                self._update_telemetry_state(risk_data)

            except Exception as e:
                print(f"[Manager] Error in main loop: {e}")
                time.sleep(0.1)

    def _get_frame(self) -> Optional[np.ndarray]:
        frame = None
        try:
            if self.tello and not self.use_webcam:
                frame_rgb = self.frame_reader.frame
                if frame_rgb is not None:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            elif self.webcam:
                ok, frame = self.webcam.read()
                if not ok:
                    frame = None
        except Exception as e:
            print(f"[Manager] Error getting frame: {e}")
            return None
        if frame is not None:
            if (
                frame.shape[0] != self.FRAME_HEIGHT
                or frame.shape[1] != self.FRAME_WIDTH
            ):
                frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        return frame

    def _generate_telemetry_snapshot(self, risk_data: Optional[Dict[str, Any]]):
        if risk_data is None:
            # 초기 상태 & 데이터 누락 시 안전 상태 반환
            risk_data = self.detector._evaluate_risk(0.0)

        return {
            "type": "telemetry",
            "connected": self.is_connected,
            "battery": self.battery_level,
            "fps": self.detector.fps,
            "risk": risk_data,
        }

    # 내부 상태 업데이트
    def _update_telemetry_state(self, risk_data: Dict[str, Any]):
        telemetry = self._generate_telemetry_snapshot(risk_data)
        with self.lock:
            self.latest_telemetry = telemetry

    # app.py에서 데이터를 가져가기 위한 인터페이스
    def get_latest_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            return self.latest_telemetry.copy()

    # 외부 인터페이스 - API
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """최신 처리된 프레임을 반환합니다 (Thread-safe)."""
        with self.lock:
            return self.latest_processed_data["frame"]

    def send_command(self, command: str, data: Optional[Dict[str, Any]] = None):
        if self.use_webcam:
            return
        if not self.tello:
            return
        try:
            if command == "rc_control":
                if data:
                    lr = int(data.get("lr", 0) * self.DRONE_SPEED)
                    fb = int(data.get("fb", 0) * self.DRONE_SPEED)
                    ud = int(data.get("ud", 0) * self.DRONE_SPEED)
                    yv = int(data.get("yv", 0) * self.DRONE_SPEED)
                    self.tello.send_rc_control(lr, fb, ud, yv)
            elif command == "takeoff":
                self.tello.takeoff()
            elif command == "land":
                self.tello.land()
        except Exception as e:
            print(f"[Manager] Failed to send command '{command}': {e}")

try:
    initial_detector = CollisionDetectorLIME(weights_path=None)
    drone_manager = DroneManager(detector=initial_detector, use_webcam=False)
except Exception as e:
    print(f"[System Init Error] Failed to initialize Detector or Manager: {e}")
    exit(1)