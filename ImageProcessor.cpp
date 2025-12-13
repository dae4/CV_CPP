#define _CRT_SECURE_NO_WARNINGS 
#include "ImageProcessor.hpp"
#include <iomanip> 
#include <ctime>   
#include <sstream> 

namespace ImageProcessor { 

    std::string getCurrentDateTime() {
        time_t now = time(0);
        struct tm* tstruct = localtime(&now);
        std::stringstream ss;
        ss << std::put_time(tstruct, "%Y-%m-%d_%H-%M-%S"); 
        return ss.str();
    }

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

    // --- [Mode 1: Motion Detector] ---
    void detectMotion(RuntimeState& state, const AppConfig& config) {
        cv::Mat processArea;

        // ROI 적용 여부 확인
        if (state.useRoi) {
            cv::Rect safeRoi = state.roiRect & cv::Rect(0, 0, state.grayFrame.cols, state.grayFrame.rows);
            if (safeRoi.area() > 0)
                processArea = state.grayFrame(safeRoi);
            else 
                processArea = state.grayFrame;
        } else {
            processArea = state.grayFrame;
        }

        if (state.prevFrame.empty() || state.prevFrame.size() != processArea.size()) {
           state.prevFrame = processArea.clone();
           return;
        }

        cv::absdiff(state.prevFrame, processArea, state.diffFrame);
        cv::threshold(state.diffFrame, state.diffFrame, config.thresholdVal, 255, cv::THRESH_BINARY); 
        cv::dilate(state.diffFrame, state.diffFrame, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(state.diffFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        state.motionRects.clear();
        for (const auto& contour : contours) { 
            if (cv::contourArea(contour) < config.minArea) continue;
            cv::Rect r = cv::boundingRect(contour);

            if (state.useRoi) {
                r.x += state.roiRect.x; 
                r.y += state.roiRect.y; 
            }
            state.motionRects.push_back(r);
        }

        // 녹화 로직
        if(!state.motionRects.empty()) {
            state.noMotionFrameCount = 0;
            if(!state.isRecording) {
                std::string fileName = "Rec_" + getCurrentDateTime() + ".avi";
                std::cout << "[Info] Recording started: " << fileName << std::endl;
                state.writer.open(fileName, cv::VideoWriter::fourcc('M','J','P','G'), 20, cv::Size(config.width, config.height));
                state.isRecording = true;
            }
        } else {
            state.noMotionFrameCount++;
            if(state.noMotionFrameCount >= 30 && state.isRecording) { 
                std::cout << "[Info] Recording stopped." << std::endl;
                state.writer.release();
                state.isRecording = false;
            }
        }
        if (state.isRecording) {
            state.writer.write(state.currentFrame);
        }

        state.prevFrame = processArea.clone();
    }

    // --- [Mode 2: Optical Flow] ---
    void computeOpticalFlow(RuntimeState& state) {
        // Optical Flow는 ROI 없이 전체 화면 사용 예시
        // prevFrame을 GrayScale 원본으로 사용 (전처리된 블러 이미지 말고)
        cv::Mat currentGray;
        cv::cvtColor(state.currentFrame, currentGray, cv::COLOR_BGR2GRAY);

        if (state.prevFrame.empty() || state.prevFrame.size() != currentGray.size()) {
            state.prevFrame = currentGray.clone();
            return;
        }

        // 간단한 Farneback Dense Optical Flow 예시
        cv::Mat flow;
        // (입력1, 입력2, 출력, scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        cv::calcOpticalFlowFarneback(state.prevFrame, currentGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // 시각화: 흐름을 격자 단위로 그림
        for (int y = 0; y < flow.rows; y += 10) {
            for (int x = 0; x < flow.cols; x += 10) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                if (fxy.x * fxy.x + fxy.y * fxy.y > 1) { // 움직임이 일정 이상일 때만
                    cv::line(state.currentFrame, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), cv::Scalar(0, 255, 0));
                    cv::circle(state.currentFrame, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        state.prevFrame = currentGray.clone();
    }
}