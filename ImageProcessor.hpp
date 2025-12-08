// ImageProcessor.hpp
#pragma once
#include "Common.hpp"

namespace ImageProcessor {
    bool initializeCamera(cv::VideoCapture& cap, const AppConfig& config);
    void preprocess(const cv::Mat& src, cv::Mat& dst);
    void detectMotion(RuntimeState& state, const AppConfig& config);
}