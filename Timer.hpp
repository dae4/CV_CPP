#pragma once
#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief Utility class for measuring FPS (Frames Per Second) and frame time.
 */
class UtilityTimer {
private:
    double freq;           // System clock frequency (ticks per second)
    int64_t lastTick;      // The tick count at the start of the last frame
    double fps;            // Current calculated FPS
    double frameTimeMs;    // Current frame time in milliseconds

public:
    UtilityTimer() : fps(0.0), frameTimeMs(0.0) {
        // Get the system clock frequency (ticks per second)
        freq = cv::getTickFrequency();
        lastTick = cv::getTickCount();
    }

    /**
     * @brief Call this at the start of every frame loop to update timing metrics.
     */
    void update() {
        int64_t currentTick = cv::getTickCount();
        
        // Calculate the number of ticks elapsed since the last frame
        int64_t elapsedTicks = currentTick - lastTick;
        
        // Calculate frame time in milliseconds
        frameTimeMs = (double)elapsedTicks * 1000.0 / freq;
        
        // Calculate FPS (1 second / elapsed time in seconds)
        if (frameTimeMs > 0) {
            fps = freq / (double)elapsedTicks;
        } else {
            fps = 0.0;
        }

        lastTick = currentTick; // Set current tick for the next frame's start
    }

    /**
     * @brief Returns the formatted FPS and ms string.
     * @return "FPS: 30.0 / 33.3 ms"
     */
    std::string getFpsString() const {
        std::string fpsStr = cv::format("%.1f", fps);
        std::string msStr = cv::format("%.1f", frameTimeMs);
        return "FPS: " + fpsStr + " / " + msStr + " ms";
    }
};