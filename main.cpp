#include <iostream>
#include "Common.hpp"
#include "ImageProcessor.hpp"
#include "DisplayManager.hpp"

// 기능별 실행 함수 선언
void runMotionDetectorMode();
void runOpticalFlowMode();

int main() {
    int choice = -1;

    while (true) {
        // --- 메인 메뉴 ---
        std::cout << "\n===================================" << std::endl;
        std::cout << "      Multi-Function CV System     " << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << " [1] Motion Detector (ROI & Rec)" << std::endl;
        std::cout << " [2] Optical Flow Visualization" << std::endl;
        std::cout << " [0] Exit" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        std::cout << " Select Mode >> ";
        std::cin >> choice;

        if (choice == 0) {
            std::cout << "Good bye!" << std::endl;
            break;
        }
        else if (choice == 1) {
            runMotionDetectorMode(); // 1번 기능 실행
        }
        else if (choice == 2) {
            runOpticalFlowMode();    // 2번 기능 실행
        }
        else {
            std::cout << "Invalid input." << std::endl;
        }
    }
    return 0;
}

// [기능 1] 기존에 작성하신 모션 디텍터 루프
void runMotionDetectorMode() {
    AppConfig config;
    RuntimeState state;
    cv::VideoCapture cap;

    config.windowName = "Mode: Motion Detector"; // 창 이름 변경

    if (!ImageProcessor::initializeCamera(cap, config)) {
        std::cerr << "Error: Camera not accessible" << std::endl;
        return;
    }

    cv::namedWindow(config.windowName);
    // 마우스 콜백 설정 (Motion Detector 전용)
    cv::setMouseCallback(config.windowName, DisplayManager::onMouse, &state);

    std::cout << ">> Motion Detector Started." << std::endl;
    std::cout << ">> Drag mouse for ROI. Press 'r' to reset ROI. Press 'q' to return menu." << std::endl;

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        // 전처리 및 모션 감지 로직 수행
        ImageProcessor::preprocess(state.currentFrame, state.grayFrame);
        ImageProcessor::detectMotion(state, config);

        DisplayManager::render(state, config);
        DisplayManager::handleInput(state); // 'q' 누르면 state.isRunning = false 됨
    }

    // 종료 처리
    if (state.isRecording) state.writer.release();
    cap.release();
    cv::destroyAllWindows();
}

// [기능 2] 옵티컬 플로우 루프
void runOpticalFlowMode() {
    AppConfig config;
    RuntimeState state;
    cv::VideoCapture cap;

    config.windowName = "Mode: Optical Flow";

    if (!ImageProcessor::initializeCamera(cap, config)) {
        std::cerr << "Error: Camera not accessible" << std::endl;
        return;
    }

    cv::namedWindow(config.windowName);
    // 옵티컬 플로우는 마우스 ROI가 필요 없다면 콜백 등록 안 함 (또는 다른 콜백 등록)

    std::cout << ">> Optical Flow Started." << std::endl;
    std::cout << ">> Press 'q' to return menu." << std::endl;

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        // 옵티컬 플로우 계산 (Render 안에 그리기 로직 포함됨)
        ImageProcessor::computeOpticalFlow(state);

        // 결과 출력 (DisplayManager의 render는 Rect를 그리는 용도라, 여기서는 단순 imshow만 해도 됨)
        // 하지만 통일성을 위해 render를 쓰되, motionRects가 비어있으므로 사각형은 안 그려짐
        cv::imshow(config.windowName, state.currentFrame);
        
        // 입력 처리
        DisplayManager::handleInput(state);
    }

    cap.release();
    cv::destroyAllWindows();
}