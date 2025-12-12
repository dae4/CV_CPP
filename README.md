# 📹 Smart Motion Detector (C++ & OpenCV)

**Smart Motion Detector**는 C++와 OpenCV를 활용하여 개발한 실시간 보안 모니터링 시스템입니다.
카메라 입력 영상에서 움직임을 실시간으로 감지하고, 사용자가 지정한 관심 영역(ROI) 내에서 이벤트가 발생할 경우 자동으로 영상을 녹화합니다.

구조체(Struct) 기반의 데이터 설계와 모듈화된 아키텍처를 적용하여 확장성과 유지보수성을 고려했습니다.

---

## ✨ Key Features (주요 기능)

* **실시간 모션 감지 (Real-time Motion Detection)**
    * 프레임 차분(Frame Differencing) 기법을 이용한 움직임 포착.
    * Gaussian Blur 및 Morphological 연산(Dilate/Erode)을 통한 노이즈 제거.
    * 설정된 임계값(Threshold) 이상의 변화만 감지.

* **관심 영역 설정 (Region of Interest - ROI)**
    * 마우스 드래그를 통해 감시하고 싶은 특정 영역만 지정 가능.
    * 영역 밖의 움직임(배경 노이즈, 불필요한 행인 등)은 무시.
    * 좌표 보정 로직을 통해 ROI 내부의 움직임을 전체 화면 좌표로 정확히 매핑.

* **스마트 자동 녹화 (Smart Auto Recording)**
    * 모션이 감지되는 순간에만 영상 저장 (저장 공간 효율화).
    * 움직임이 멈추더라도 일정 시간(Buffer) 대기 후 녹화 종료.
    * 파일명 자동 생성: `Rec_YYYY-MM-DD_HH-MM-SS.avi` 포맷 적용.

* **모듈화된 아키텍처 (Modular Architecture)**
    * **Common**: 설정(`AppConfig`)과 상태(`RuntimeState`) 데이터 구조체 정의.
    * **ImageProcessor**: OpenCV 영상 처리 알고리즘 전담.
    * **DisplayManager**: UI 렌더링 및 사용자 입력 처리.

---

## 🛠 Tech Stack

* **Language**: C++ (C++14 Standard)
* **Library**: OpenCV 4.x
* **Build System**: CMake

---

## 🚀 How to Run (실행 방법)

### Prerequisites
* C++ 컴파일러 (g++, clang, or MSVC)
* OpenCV 라이브러리 설치 필요
* CMake 3.10 이상

### Build & Execute
```bash
# 1. 프로젝트 클론 및 폴더 이동
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_FOLDER>

# 2. 빌드 디렉토리 생성
mkdir build
cd build

# 3. CMake 설정 및 빌드
cmake ..
make

# 4. 실행
./MotionDetector