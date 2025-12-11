// ImageProcessor.cpp
#define _CRT_SECURE_NO_WARNINGS // localtime 사용 시 에러 방지 
#include "ImageProcessor.hpp"
#include <iomanip> // 날짜 포맷팅용 (put_time)
#include <ctime>   // 시간 가져오기용
#include <sstream> // 문자열 조합용

namespace ImageProcessor { 


    std::string getCurrentDateTime() {
        time_t now = time(0);
        struct tm* tstruct = localtime(&now);
        
        std::stringstream ss;
        // %Y:년도, %m:월, %d:일, %H:시, %M:분, %S:초
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
        cv::GaussianBlur(dst, dst, cv::Size(21, 21), 0); // 노이즈 감소
    }

    void detectMotion(RuntimeState& state, const AppConfig& config) {
        cv::Mat processArea;

        if (state.useRoi) {
            // 안전한 ROI 사각형 계산 (이미지 밖으로 나가는 것 방지)
            cv::Rect safeRoi = state.roiRect & cv::Rect(0, 0, state.grayFrame.cols, state.grayFrame.rows);
            processArea = state.grayFrame(safeRoi); // 이미지를 자름 (참조만 생성, 복사 아님)
        } else {
            processArea = state.grayFrame; // 전체 이미지
        }

        if (state.prevFrame.empty() || state.prevFrame.size() != processArea.size()) {
           state.prevFrame = processArea.clone();
            return;
        }

        cv::absdiff(state.prevFrame, processArea, state.diffFrame); // 프레임 차이 계산
        cv::threshold(state.diffFrame, state.diffFrame, config.thresholdVal, 255, cv::THRESH_BINARY); 
        cv::dilate(state.diffFrame, state.diffFrame, cv::Mat(), cv::Point(-1, -1), 2); // 팽창 연산으로 노이즈 제거

        std::vector<std::vector<cv::Point>> contours;  // 윤곽선 검출
        cv::findContours(state.diffFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        state.motionRects.clear();
        for (const auto& contour : contours) { //auto : Datatype 추론
            if (cv::contourArea(contour) < config.minArea) continue;
            cv::Rect r = cv::boundingRect(contour);

            // 잘린 이미지(processArea) 기준의 좌표(r)를
            // 전체 화면 기준의 좌표로 바꿔줘야 화면에 제대로 그려짐
            if (state.useRoi) {
                r.x += state.roiRect.x; // ROI 시작점만큼 x 이동
                r.y += state.roiRect.y; // ROI 시작점만큼 y 이동
            }
            state.motionRects.push_back(r);
        }

        // isRecording
        if(!state.motionRects.empty()) {
            state.noMotionFrameCount = 0;
            if(!state.isRecording) {
                std::string fileName = "Rec_" + getCurrentDateTime() + ".avi";
                
                std::cout << "Recording started: " << fileName << std::endl;

                state.writer.open(fileName, 
                              cv::VideoWriter::fourcc('M','J','P','G'), 
                              20, 
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

        state.prevFrame = processArea.clone();
    }
}