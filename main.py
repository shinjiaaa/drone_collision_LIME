# app.py
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

# detector.py에 네가 준 CollisionDetectorLIME 클래스가 있다고 가정
from system.detector import CollisionDetectorLIME

app = FastAPI(title="LIME Collision Detector - Upload Demo")

# 정적(간단) 페이지 제공을 위해 임시 디렉토리 사용 (필요시 변경)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# 단일 Detector 인스턴스: 서버가 띄워질 때 모델 로드(비용 큼)
detector = CollisionDetectorLIME()  # 필요하면 경로 인자 전달

# 간단한 업로드/미리보기용 HTML (drag & drop)
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Upload Image / Video - CollisionDetectorLIME</title>
  <style>
    body{font-family: Arial, Helvetica, sans-serif; margin:40px}
    .drop{border:2px dashed #888; padding:30px; border-radius:8px; text-align:center}
    #preview{margin-top:20px; max-width:90%; height:auto; display:block}
    .btn{margin-top:10px}
  </style>
</head>
<body>
  <h2>Drag & Drop Image or Video</h2>
  <div class="drop" id="dropzone">Drop files here or <input id="file" type="file" /></div>
  <div>
    <button onclick="upload('image')" class="btn">Upload as Image</button>
    <button onclick="upload('video')" class="btn">Upload as Video</button>
  </div>
  <p id="status"></p>
  <img id="preview" src="" alt="" />
  <video id="vpreview" controls style="display:none; max-width:90%"></video>

<script>
const drop = document.getElementById('dropzone');
const fileInput = document.getElementById('file');
let file = null;
drop.addEventListener('dragover', e => { e.preventDefault(); drop.style.borderColor='#44a'; });
drop.addEventListener('dragleave', e => { drop.style.borderColor='#888'; });
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.style.borderColor='#888';
  file = e.dataTransfer.files[0];
  showPreview(file);
});
fileInput.addEventListener('change', e => { file = e.target.files[0]; showPreview(file); });

function showPreview(f){
  const status = document.getElementById('status');
  status.textContent = '';
  const img = document.getElementById('preview');
  const v = document.getElementById('vpreview');
  img.style.display='none'; v.style.display='none';
  if(!f) return;
  const url = URL.createObjectURL(f);
  if(f.type.startsWith('image/')){ img.src=url; img.style.display='block'; }
  else if(f.type.startsWith('video/')){ v.src=url; v.style.display='block'; }
  else { status.textContent='지원되지 않는 파일 형식입니다.'; }
}

async function upload(kind){
  const status = document.getElementById('status');
  if(!file){ status.textContent='먼저 파일을 선택 또는 드래그하세요.'; return; }
  status.textContent='업로드 중...';
  const fd = new FormData();
  fd.append('file', file);
  const endpoint = kind === 'image' ? '/process/image' : '/process/video';
  try{
    const res = await fetch(endpoint, { method:'POST', body: fd });
    if(!res.ok){
      const txt = await res.text();
      status.textContent = '서버 에러: ' + txt;
      return;
    }
    // 이미지: image/jpeg, 비디오: video/mp4
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    if(kind === 'image'){
      document.getElementById('preview').src = url;
      document.getElementById('preview').style.display='block';
      document.getElementById('vpreview').style.display='none';
    } else {
      const v = document.getElementById('vpreview');
      v.src = url; v.style.display='block';
      document.getElementById('preview').style.display='none';
    }
    status.textContent='처리 완료';
  }catch(e){
    status.textContent='네트워크 에러: ' + e;
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


# Helper: read image bytes to BGR numpy
def read_imagefile_to_bgr(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# 이미지 업로드 처리: 단일 이미지 반환 (JPEG)
@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "image 파일이 필요함"}, status_code=400)
    data = await file.read()
    img = read_imagefile_to_bgr(data)
    if img is None:
        return JSONResponse({"error": "이미지 디코딩 실패"}, status_code=400)

    # 처리 (동기 함수라서 CPU-bound; 짧은 처리에선 ok)
    processed, info = detector.process_frame(img)

    # encode to JPEG
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        return JSONResponse({"error": "인코딩 실패"}, status_code=500)
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


# 비디오 업로드 처리: MP4 반환 (프레임 단위로 detector.process_frame)
@app.post("/process/video")
async def process_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return JSONResponse({"error": "video 파일이 필요함"}, status_code=400)

    # 임시 파일로 저장
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

        # 비디오 읽기
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "비디오 열기 실패"}, status_code=400)

        # 출력 설정 (same size as input, mp4v)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # 프레임 단위 처리 (주의: 느릴 수 있음 — 필요시 분해능 축소)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, info = detector.process_frame(frame)
            # processed_frame는 BGR
            writer.write(processed_frame)

        cap.release()
        writer.release()

        # 결과 파일 반환
        return FileResponse(out_path, media_type="video/mp4", filename="processed.mp4")
    finally:
        # 클린업: 임시 업로드 파일은 삭제
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # 디버그용 로컬 실행: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")