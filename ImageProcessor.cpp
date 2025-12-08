// ImageProcessor.cpp
#include "ImageProcessor.hpp"

namespace ImageProcessor {

    bool initializeCamera(cv::VideoCapture& cap, const AppConfig& config) {
        cap.open(config.deviceID);
        if (!cap.isOpened()) return false;
        
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
        return true;
    }

    void preprocess(const cv::Mat& src, cv::Mat& dst) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(dst, dst, cv::Size(21, 21), 0);
    }

    void detectMotion(RuntimeState& state, const AppConfig& config) {
        if (state.prevFrame.empty()) {
            state.prevFrame = state.grayFrame.clone();
            return;
        }

        cv::absdiff(state.prevFrame, state.grayFrame, state.diffFrame);
        cv::threshold(state.diffFrame, state.diffFrame, config.thresholdVal, 255, cv::THRESH_BINARY);
        cv::dilate(state.diffFrame, state.diffFrame, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(state.diffFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        state.motionRects.clear();
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < config.minArea) continue;
            state.motionRects.push_back(cv::boundingRect(contour));
        }

        state.prevFrame = state.grayFrame.clone();
    }
}