# drone_collision_LIME
2025 KSC - Application of the LIME Technique for Real-Time Drone Collision Risk Prediction

## dataset
객체 탐지 모델
- https://universe.roboflow.com/tylervisimoai/drone-crash-avoidance
- https://github.com/VisDrone/VisDrone-Dataset
충돌 분류 모델
- https://github.com/uzh-rpg/rpg_public_dronet

## model route
객체 탐지 모델: models/best.pt
충돌 분류 모델: models/model_weights.h5

## Directory
static/
    app/: 실시간 영상 프레임 처리 시스템
    insert_image/: image & video 처리 시스템
system/: 내부 기능 구현
models/: 모델
train/: 각 모델 학습
eval/: 각 모델 평가