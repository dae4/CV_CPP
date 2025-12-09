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
        cv::GaussianBlur(dst, dst, cv::Size(21, 21), 0); // 노이즈 감소
    }

    void detectMotion(RuntimeState& state, const AppConfig& config) {
        if (state.prevFrame.empty()) {
           state.prevFrame = state.grayFrame.clone();
            return;
        }
        // isRecording
        if(!state.motionRects.empty()) {
            state.noMotionFrameCount = 0;
            if(!state.isRecording) {
                state.writer.open("output_"+ std::to_string(time(0))+".avi", cv::VideoWriter::fourcc('M','J','P','G'), 20,
                                  cv::Size(config.width, config.height));
                state.isRecording = true;
            }
        } else {
            state.noMotionFrameCount++;
            if(state.noMotionFrameCount >= 30 && state.isRecording) { // 30 프레임 이상 움직임 없으면 녹화 중지
                state.writer.release();
                state.isRecording = false;
            }
        }
        if (state.isRecording) {
            state.writer.write(state.currentFrame);
        }
        cv::absdiff(state.prevFrame, state.grayFrame, state.diffFrame); // 프레임 차이 계산
        cv::threshold(state.diffFrame, state.diffFrame, config.thresholdVal, 255, cv::THRESH_BINARY); 
        cv::dilate(state.diffFrame, state.diffFrame, cv::Mat(), cv::Point(-1, -1), 2); // 팽창 연산으로 노이즈 제거

        std::vector<std::vector<cv::Point>> contours;  // 윤곽선 검출
        cv::findContours(state.diffFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        state.motionRects.clear();
        for (const auto& contour : contours) { //auto : Datatype 추론
            if (cv::contourArea(contour) < config.minArea) continue;
            state.motionRects.push_back(cv::boundingRect(contour));
        }

        state.prevFrame = state.grayFrame.clone();
    }
}