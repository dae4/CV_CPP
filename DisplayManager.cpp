// DisplayManager.cpp
#include "DisplayManager.hpp"
#include <iostream> // for std::cout if needed

namespace DisplayManager {

    void onMouse(int event, int x, int y, int flags, void* userdata) {
        RuntimeState* state = (RuntimeState*)userdata;

        if (event == cv::EVENT_LBUTTONDOWN) {
            //Record starting point and set dragging flag
            state->isDragging = true;
            state->startPoint = cv::Point(x, y);
            state->useRoi = false; //Reset ROI usage until drag is complete
        } 
        else if (event == cv::EVENT_MOUSEMOVE) {
            //Mouse move: update current point if dragging
            if (state->isDragging) {
                state->curPoint = cv::Point(x, y);
            }
        } 
        else if (event == cv::EVENT_LBUTTONUP) {
            //Mouse release: finalize ROI
            state->isDragging = false;
            
            
            cv::Rect rect(state->startPoint, cv::Point(x, y));
            
            // if width height is too small, ignore 
            if (rect.width > 10 && rect.height > 10) {
                state->roiRect = rect;
                state->useRoi = true; 
                
                // Reset previous frame to avoid artifacts
                state->prevFrame = cv::Mat(); 
            }
        }
    }

    void render(const RuntimeState& state, const AppConfig& config) {
        cv::Mat display = state.currentFrame.clone();

        // Draw motion rectangles
        for (const auto& rect : state.motionRects) {
            cv::rectangle(display, rect, cv::Scalar(0, 0, 255), 2);
        }

        if (state.useRoi) {
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
