// DisplayManager.cpp
#include "DisplayManager.hpp"
#include <iostream> // for std::cout if needed

namespace DisplayManager {
    void render(const RuntimeState& state, const AppConfig& config) {
        cv::Mat display = state.currentFrame.clone();
        
        for (const auto& rect : state.motionRects) {
            cv::rectangle(display, rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(display, "Motion Detected", cv::Point(10, 20), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow(config.windowName, display);
    }

    void handleInput(RuntimeState& state) {
        char key = (char)cv::waitKey(30);
        if (key == 27 || key == 'q') { 
            state.isRunning = false;
        }
    }
}