from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import base64
from system.static_detector import CollisionDetectorImage

app = FastAPI()
detector = CollisionDetectorImage()


def encode_image_to_base64(frame):
    import cv2, numpy as np, base64

    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    frame, risk_data, mask_pos = detector.process_image(img_bytes)
    if frame is None:
        return JSONResponse({"error": "이미지 처리 실패"})

    img_base64 = encode_image_to_base64(frame)
    mask_base64 = (
        encode_image_to_base64((mask_pos * 255).astype("uint8"))
        if mask_pos is not None
        else None
    )

    return JSONResponse({"image": img_base64, "mask": mask_base64, "risk": risk_data})
