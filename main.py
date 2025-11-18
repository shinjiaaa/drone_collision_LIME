import io
import os
import tempfile
import asyncio
import json
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
import base64

# ----------------------------
# LIME 자연어 생성 함수 import
# ----------------------------
from system.explainer import generate_lime_explanation

# ----------------------------
# Detector
# ----------------------------
from system.static_detector import StaticCollisionDetectorLIME

app = FastAPI(title="LIME Collision Detector - Upload Demo")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

detector = StaticCollisionDetectorLIME()


# ★ 이미지 파일 → numpy BGR
def read_imagefile_to_bgr(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        """
    <h3>Upload Image</h3>
    <form action="/process/image" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Upload</button>
    </form>
    """
    )


# ----------------------------------------------------
#  이미지 업로드 처리 + LIME 자연어 설명 결합
# ----------------------------------------------------
import base64
import json
from fastapi.responses import HTMLResponse


@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    data = await file.read()
    img = read_imagefile_to_bgr(data)
    if img is None:
        return HTMLResponse("<p>이미지 디코딩 실패</p>", status_code=400)

    processed, info = detector.process_frame(img)

    # info에서 LIME 관련 값 가져오기
    pos_mask = info.get("pos_mask")
    neg_mask = info.get("neg_mask")
    class_name = info.get("class_name", "unknown")
    collision_prob = info.get("collision_prob", 0.0)

    explanation = generate_lime_explanation(
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        class_name=class_name,
        collision_prob=collision_prob,
    )

    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        return HTMLResponse("<p>이미지 인코딩 실패</p>", status_code=500)

    img_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    explanation_text = json.dumps(explanation, ensure_ascii=False, indent=2)

    html_content = f"""
    <h3>Processed Image</h3>
    <img src="data:image/jpeg;base64,{img_base64}" />
    <h3>LIME Explanation</h3>
    <pre>{explanation_text}</pre>
    <a href="/">Back</a>
    """
    return HTMLResponse(html_content)


# ----------------------------------------------------
#  비디오 업로드 처리
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
