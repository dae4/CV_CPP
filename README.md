# 👁️ Smart Computer Vision System (C++ & OpenCV)
Smart Computer Vision System은 C++와 OpenCV를 활용하여 개발한 올인원 영상 처리 애플리케이션입니다. 기존의 모션 감지 CCTV 기능을 넘어, 딥러닝 기반의 얼굴 인식(Privacy), 객체 추적(Tracking), 그리고 증강현실(AR) 드로잉 기능까지 하나의 통합된 시스템에서 제공합니다.


## ✨ Key Features (주요 기능)
이 프로젝트는 5가지의 독립적인 모드를 제공합니다.

1. 📹 Smart Motion Detector (지능형 모션 감지)
    * Real-time Detection: 프레임 차분(Frame Differencing) 기법을 이용한 실시간 움직임 포착.

    * Smart ROI: 마우스 드래그로 지정한 **관심 영역(ROI)**만 감시하며, 영역 밖의 노이즈는 무시합니다.

    * Auto Recording: 움직임이 감지되는 순간에만 영상을 .avi 파일로 자동 녹화하여 저장 공간을 효율적으로 사용합니다.

2. 🌊 Optical Flow (광학 흐름 시각화)
    * Dense Optical Flow: Farneback 알고리즘을 사용하여 화면 내 모든 픽셀의 움직임 벡터를 계산합니다.

    * Visualization: 물체의 이동 방향과 속도를 격자(Grid) 형태의 벡터로 시각화하여 흐름을 분석합니다.

3. 🛡️ AI Face Blur (프라이버시 보호)
    * Deep Learning (DNN): OpenCV DNN 모듈과 ResNet10(SSD) 모델을 사용하여 높은 정확도로 얼굴을 인식합니다.

    * Privacy Masking: 인식된 얼굴 영역을 자동으로 블러(Blur) 처리하여 실시간으로 익명화를 수행합니다.

    * Robustness: 측면 얼굴이나 다양한 조명 환경에서도 기존 Haar Cascade보다 월등한 인식률을 보입니다.

4. 🎯 Object Tracker (객체 추적)
    * CSRT Tracker: 정확도가 높은 CSRT 알고리즘을 적용하여 사용자가 지정한 객체를 끈질기게 추적합니다.

    * Occlusion Handling: 물체가 잠시 가려지거나 이동 경로가 불규칙해도 추적을 유지합니다.

    * Mouse Interaction: 마우스 드래그로 추적 대상을 직관적으로 설정할 수 있습니다.

5. 🎨 Air Canvas (허공에 그림 그리기)
    * Color Tracking: HSV 색상 공간을 활용하여 파란색(Blue) 물체를 펜으로 인식합니다.

    * Noise Reduction: 모폴로지 연산(침식/팽창)을 통해 노이즈를 제거하고 깔끔한 선을 그립니다.

    * Interactive Drawing: 허공에 제스처를 취해 그림을 그리고, 키보드로 캔버스를 초기화합니다.

6. 🧠 YOLOv8 Object Detector (실시간 사물 인식)
    * State-of-the-Art Model: 경량화된 YOLOv8n (Nano) 모델을 사용하여 실시간 객체 탐지.

    * YOLOv8.onnx 사용하여 객체탐지.

## 🛠 Tech Stack
    Language: C++ (C++14 Standard)

    Library: OpenCV 4.x (Core, ImgProc, Video, DNN, Tracking modules)

    Build System: CMake 3.10+

    Environment: Windows / Linux / macOS (Cross-Platform Compatible)

## 🚀 How to Run (실행 방법)
1. Prerequisites (필수 준비물)
    deploy.prototxt

    res10_300x300_ssd_iter_140000.caffemodel

    (OpenCV 공식 GitHub의 samples/data/dnn/face_detector 경로에서 다운로드 가능)

    YOLOv8 yolov8n.onnx, classes.txt

2. Build & Execute
``` Bash

# 1. 프로젝트 클론
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_FOLDER>

# 2. 빌드 디렉토리 생성
mkdir build
cd build

# 3. CMake 설정 및 빌드
cmake ..
make 
# Windows의 경우: cmake --build . --config Release

# 4. 실행
./CV_application
```