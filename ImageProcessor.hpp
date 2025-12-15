#pragma once
#include "Common.hpp"

namespace ImageProcessor {
    // Utility: Initializes camera with config settings
    bool initializeCamera(cv::VideoCapture& cap, const AppConfig& config);
    
    // Utility: Pre-processing (Grayscale, Blur)
    void preprocess(const cv::Mat& src, cv::Mat& dst);
    
    // Utility: Render FPS on frame
    void renderFPS(cv::Mat& frame, const std::string& fpsString);


    // [Feature 1] Motion Detection Algorithm
    void detectMotion(RuntimeState& state, const AppConfig& config);

    // [Feature 2] Optical Flow Visualization
    void computeOpticalFlow(RuntimeState& state);

    // [Feature 3] Face Detection & Anonymization (Blur)
    void processFaceBlur(RuntimeState& state);

    // [Feature 4] Object Tracking (KCF Algorithm)
    void processObjectTracking(RuntimeState& state);

    // [Feature 5] Air Canvas (Color Tracking)
    void processAirCanvas(RuntimeState& state, const AppConfig& config);

    // [Feature 6] YOLO Object Detection
    void processYOLODetection(RuntimeState& state, const AppConfig& config);

    
}