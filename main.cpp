// main.cpp
#include <iostream>
#include "Common.hpp"
#include "ImageProcessor.hpp"
#include "DisplayManager.hpp"

int main() {
    AppConfig config;
    RuntimeState state;
    cv::VideoCapture cap;

    if (!ImageProcessor::initializeCamera(cap, config)) {
        std::cerr << "Error: Camera not accessible" << std::endl;
        return -1;
    }

    cv::namedWindow(config.windowName);
    cv::setMouseCallback(config.windowName, DisplayManager::onMouse, &state);

    std::cout << "Drag mouse to set ROI. Press 'r' to reset." << std::endl;
    std::cout << "System Started. Press 'q' to exit." << std::endl;

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::preprocess(state.currentFrame, state.grayFrame);
        ImageProcessor::detectMotion(state, config);

        DisplayManager::render(state, config);
        DisplayManager::handleInput(state);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}