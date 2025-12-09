// Common.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// 설정 데이터
struct AppConfig {
    int deviceID = 0;
    int width = 640;
    int height = 480;
    int thresholdVal = 30;
    int minArea = 500;
    std::string windowName = "Motion Detector Engine";
};

// 런타임 상태 데이터
struct RuntimeState {
    cv::Mat currentFrame;
    cv::Mat grayFrame;
    cv::Mat prevFrame;
    cv::Mat diffFrame;
    std::vector<cv::Rect> motionRects;
    bool isRunning = true;
    cv::VideoWriter writer;
    bool isRecording = false;
    int noMotionFrameCount = 0;
};