#pragma once
#include "Common.hpp"

namespace ImageProcessor {
    std::string getCurrentDateTime();
    bool initializeCamera(cv::VideoCapture& cap, const AppConfig& config);
    void preprocess(const cv::Mat& src, cv::Mat& dst);

    // [Mode 1] 모션 감지 & 녹화
    void detectMotion(RuntimeState& state, const AppConfig& config);

    // [Mode 2] 옵티컬 플로우 (새로 추가됨)
    void computeOpticalFlow(RuntimeState& state);
}