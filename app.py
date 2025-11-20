# app.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
import asyncio
import uvicorn
import cv2
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from contextlib import asynccontextmanager
from typing import Dict, Any

# drone_manager.py에서 싱글톤 인스턴스 임포트
try:
    from system.drone_manager import drone_manager
except ImportError as e:
    print(f"[ERROR] drone_manager.py 또는 detector.py를 찾을 수 없습니다. {e}")
    exit()


# --- 서버 시작/종료 시점 관리 (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 (Startup)
    print("[Server] Starting up...")
    asyncio.create_task(connect_drone_async())
    # [수정됨] 텔레메트리 전송 백그라운드 루프 시작
    broadcast_task = asyncio.create_task(broadcast_telemetry_loop())
    yield
    # 서버 종료 시 (Shutdown)
    print("[Server] Shutting down...")
    # 백그라운드 루프 안전하게 종료
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    drone_manager.shutdown()


async def connect_drone_async():
    """비동기로 드론 연결을 시도합니다."""
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, drone_manager.connect)
    if not success:
        print("[Server] Failed to connect to Drone/Webcam.")


app = FastAPI(lifespan=lifespan)

# 정적 파일 서빙
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    print("[WARN] 'static' directory not found.")


# --- 웹소켓 연결 관리 ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        # 데이터 직렬화는 한 번만 수행
        message_str = json.dumps(message)

        # 연결이 끊어진 클라이언트를 대비하여 안전하게 순회
        connections_to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception:
                connections_to_remove.append(connection)

        for connection in connections_to_remove:
            self.disconnect(connection)


manager = ConnectionManager()


# [신규] 텔레메트리 전송 루프 (Pull 방식)
async def broadcast_telemetry_loop():
    """주기적으로 DroneManager에서 최신 데이터를 가져와 브로드캐스트합니다."""
    print("[Server] Telemetry broadcast loop started.")
    while True:
        try:
            # 최신 데이터 가져오기 (Thread-safe)
            telemetry_data = drone_manager.get_latest_telemetry()
            # 모든 클라이언트에게 전송
            await manager.broadcast(telemetry_data)

            # 전송 주기 설정 (10Hz = 0.1초)
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("[Server] Telemetry broadcast loop cancelled.")
            break
        except Exception as e:
            print(f"[Server] Error in telemetry loop: {e}")
            await asyncio.sleep(1.0)


# (이전 버전의 telemetry_callback 함수 및 할당 코드는 제거됨)


# --- API 엔드포인트 ---


@app.get("/")
async def get():
    # (이전과 동일)
    try:
        with open("static/app/front/index.html", encoding='utf-8') as f:
            html = f.read()
        return HTMLResponse(html)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: static/app/front/index.html not found</h1>", status_code=404)

# MJPEG 스트리밍 구현
async def generate_mjpeg_stream():
    # (이전과 동일)
    while True:
        frame = drone_manager.get_latest_frame()
        if frame is not None:
            ret, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        await asyncio.sleep(0.033)

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import base64
import numpy as np
import cv2


detector = drone_manager.detector
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """YOLO + Collision + LIME 파이프라인 → base64로 반환"""

    if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
        return {"error": "Only PNG, JPG images are allowed."}

    # 이미지 읽기
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format."}

    # YOLO + Collision + LIME 전체 처리
    output_frame, risk_data = detector.process_frame(img)

    # output_frame → base64 JPG
    ok, buffer = cv2.imencode(".jpg", output_frame)
    if not ok:
        return {"error": "Failed to process final frame."}
    result_b64 = base64.b64encode(buffer).decode()

    # ------------------------------------------
    # 🔥 여기 추가됨: latest_lime_mask_img 변환
    # ------------------------------------------
    lime_mask_img = detector.latest_lime_mask_img  # ← 이름 맞춰 반영

    if lime_mask_img is not None:
        ok2, limbuf = cv2.imencode(".jpg", lime_mask_img)
        lime_mask_base64 = base64.b64encode(limbuf).decode() if ok2 else None
    else:
        lime_mask_base64 = None
    # ------------------------------------------

    return JSONResponse({
        "result_img": f"data:image/jpeg;base64,{result_b64}",
        "lime_mask_img": f"data:image/jpeg;base64,{lime_mask_base64}" if lime_mask_base64 else None,
        "risk_level": risk_data.get("level"),
        "max_conf": risk_data.get("max_conf")
    })


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    """제어 명령 수신용 웹소켓."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_message(message)
            except json.JSONDecodeError:
                print(f"[Server] Received invalid JSON: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("[Server] Client disconnected from /ws/data")
    except Exception as e:
        print(f"[Server] Error in websocket handler: {e}")
        manager.disconnect(websocket)


async def handle_message(message: Dict[str, Any]):
    """클라이언트로부터 수신된 메시지 처리 (제어 명령)."""
    msg_type = message.get("type")

    if msg_type == "control_command":
        command = message.get("command")
        data = message.get("data")
        if command:
            # 블로킹 I/O인 send_command를 별도 스레드에서 실행
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, drone_manager.send_command, command, data)


if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
