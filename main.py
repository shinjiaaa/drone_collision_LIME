# main.py
import io
import os
import tempfile
import asyncio
from typing import List

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np
import uvicorn
import aiofiles

# ----------------------------
# LIME 자연어 생성 함수 import
# ----------------------------
from system.lime_explainer import generate_lime_explanation

# ----------------------------
# Detector
# ----------------------------
from system.static_detector import CollisionDetectorLIME

app = FastAPI(title="LIME Collision Detector - Upload Demo")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

detector = CollisionDetectorLIME()


# ★ 이미지 파일 → numpy BGR
def read_imagefile_to_bgr(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("<h3>Upload Endpoint Ready</h3>")


# ----------------------------------------------------
#  이미지 업로드 처리 + LIME 자연어 설명 결합
# ----------------------------------------------------
@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "image 파일이 필요함"}, status_code=400)

    data = await file.read()
    img = read_imagefile_to_bgr(data)
    if img is None:
        return JSONResponse({"error": "이미지 디코딩 실패"}, status_code=400)

    # detector.process_frame → (처리된이미지, info)
    processed, info = detector.process_frame(img)

    # ---------------------------------------------
    #   info 내부 구조 예시 (네 구조 그대로 사용)
    #   info = {
    #       "pos_mask": np.ndarray,
    #       "neg_mask": np.ndarray,
    #       "class_name": str,
    #       "collision_prob": float
    #   }
    # ---------------------------------------------

    # LIME 설명 생성 실행 🔥
    explanation = generate_lime_explanation(
        pos_mask=info["pos_mask"],
        neg_mask=info["neg_mask"],
        class_name=info["class_name"],
        collision_prob=info["collision_prob"],
    )

    # 이미지 JPEG 인코딩
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        return JSONResponse({"error": "이미지 인코딩 실패"}, status_code=500)

    # 클라이언트가 JSON + 이미지 둘 다 필요하면?
    # → multipart response 사용
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers={"X-LIME-Explanation": json.dumps(explanation, ensure_ascii=False)},
    )


# ----------------------------------------------------
#  비디오 업로드 처리 (LIME 설명은 프레임마다 생성 X)
# ----------------------------------------------------
@app.post("/process/video")
async def process_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse({"error": "video 파일이 필요함"}, status_code=400)

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=os.path.splitext(file.filename)[1] or ".mp4"
    )
    os.close(tmp_fd)

    try:
        async with aiofiles.open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out.write(chunk)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "비디오 열기 실패"}, status_code=400)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, info = detector.process_frame(frame)
            writer.write(processed_frame)

        cap.release()
        writer.release()

        return FileResponse(out_path, media_type="video/mp4", filename="processed.mp4")

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
