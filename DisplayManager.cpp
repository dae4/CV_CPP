// DisplayManager.cpp
#include "DisplayManager.hpp"
#include <iostream> // for std::cout if needed

namespace DisplayManager {

    void onMouse(int event, int x, int y, int flags, void* userdata) {
        // void* 형변환: "이거 원래 RuntimeState 였어" 라고 알려줌
        RuntimeState* state = (RuntimeState*)userdata;

        if (event == cv::EVENT_LBUTTONDOWN) {
            // 클릭 시작: 드래그 시작 알림, 시작점 기록
            state->isDragging = true;
            state->startPoint = cv::Point(x, y);
            state->useRoi = false; // 새로 그리는 중이니 기존 ROI 해제
        } 
        else if (event == cv::EVENT_MOUSEMOVE) {
            // 마우스 이동: 드래그 중이라면 현재 위치 계속 업데이트
            if (state->isDragging) {
                state->curPoint = cv::Point(x, y);
            }
        } 
        else if (event == cv::EVENT_LBUTTONUP) {
            // 클릭 뗌: 드래그 종료, 최종 사각형 확정
            state->isDragging = false;
            
            // 시작점(startPoint)과 끝점(x,y)으로 사각형 생성
            cv::Rect rect(state->startPoint, cv::Point(x, y));
            
            // 너무 작은 점 같은 클릭은 무시 (가로세로 10픽셀 이상만 인정)
            if (rect.width > 10 && rect.height > 10) {
                state->roiRect = rect;
                state->useRoi = true; // 이제부터 이 영역만 감시!
                
                // [중요] 모션 감지 로직 리셋 (화면 크기가 바뀌므로 이전 프레임 삭제 필요)
                state->prevFrame = cv::Mat(); 
            }
        }
    }

    void render(const RuntimeState& state, const AppConfig& config) {
        cv::Mat display = state.currentFrame.clone();
        
        for (const auto& rect : state.motionRects) {
            cv::rectangle(display, rect, cv::Scalar(0, 0, 255), 2); // 빨간색
        }

        if (state.useRoi) {
            // 확정된 영역 (초록색)
            cv::rectangle(display, state.roiRect, cv::Scalar(0, 255, 0), 2);
            cv::putText(display, "ROI Active", state.roiRect.tl() - cv::Point(0, 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        if (state.isDragging) {
            cv::rectangle(display, cv::Rect(state.startPoint, state.curPoint), cv::Scalar(0, 255, 255), 1);
        }

        cv::imshow(config.windowName, display);
    }
    void handleInput(RuntimeState& state) {
        char key = (char)cv::waitKey(30);
        if (key == 27 || key == 'q') { 
            state.isRunning = false;
        }
        if (key == 'r') { //reset
            state.useRoi = false;
            state.prevFrame = cv::Mat(); 
        }
    }
}
