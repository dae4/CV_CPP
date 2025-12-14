#pragma once
#include "Common.hpp"
#include "ImageProcessor.hpp"
#include "DisplayManager.hpp"
#include <iostream>

class AppController {
public:
    AppController();
    void run(); // Main Menu Loop

private:
    // Shared Resources
    AppConfig config;
    RuntimeState state;
    cv::VideoCapture cap;

    // Helper: Initialize Camera & Window
    bool setupMode(const std::string& modeName);
    
    // Mode Executors (Moved from main.cpp)
    void executeMotionMode();
    void executeOpticalMode();
    void executeFaceBlurMode();
    void executeTrackerMode();
    void executeCanvasMode();
};