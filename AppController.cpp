#include "AppController.hpp"

AppController::AppController() {
    // Default config initialization if needed
}

void AppController::run() {
    int choice = -1;
    while (true) {
        std::cout << "\n=== CV Project Controller ===" << std::endl;
        std::cout << " 1. Motion Detector" << std::endl;
        std::cout << " 2. Optical Flow" << std::endl;
        std::cout << " 3. Face Blur" << std::endl;
        std::cout << " 4. Object Tracker" << std::endl;
        std::cout << " 5. Air Canvas" << std::endl;
        std::cout << " 0. Exit" << std::endl;
        std::cout << " Select >> ";
        std::cin >> choice;

        if (std::cin.fail()) {
            std::cin.clear(); std::cin.ignore(1000, '\n'); continue;
        }
        if (choice == 0) break;

        // Route to specific mode execution
        switch (choice) {
            case 1: executeMotionMode(); break;
            case 2: executeOpticalMode(); break;
            case 3: executeFaceBlurMode(); break;
            case 4: executeTrackerMode(); break;
            case 5: executeCanvasMode(); break;
            default: std::cout << "Invalid Selection." << std::endl;
        }
    }
}

// Common Setup Helper
bool AppController::setupMode(const std::string& modeName) {
    config.windowName = modeName;
    if (!ImageProcessor::initializeCamera(cap, config)) {
        std::cerr << "[Error] Camera Init Failed." << std::endl;
        return false;
    }
    cv::namedWindow(config.windowName);
    
    // Reset State for fresh start
    state.isRunning = true;
    state.useRoi = false;
    state.isTracking = false;
    state.motionRects.clear();
    state.canvasStrokes.clear();
    state.currentStroke.clear();
    state.prevFrame.release();
    
    return true;
}

// --- Specific Mode Implementations ---

void AppController::executeMotionMode() {
    if (!setupMode("Mode: Motion Detector")) return;
    cv::setMouseCallback(config.windowName, DisplayManager::onMouse, &state);

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::preprocess(state.currentFrame, state.grayFrame);
        ImageProcessor::detectMotion(state, config); // Algo Call

        DisplayManager::render(state, config);       // UI Call
        DisplayManager::handleInput(state);          // Input Call
    }
    if (state.isRecording) state.writer.release();
    cap.release();
    cv::destroyAllWindows();
}

void AppController::executeOpticalMode() {
    if (!setupMode("Mode: Optical Flow")) return;

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::computeOpticalFlow(state); // Algo Call

        cv::imshow(config.windowName, state.currentFrame);
        DisplayManager::handleInput(state);
    }
    cap.release();
    cv::destroyAllWindows();
}

void AppController::executeFaceBlurMode() {
    if (!setupMode("Mode: Face Blur")) return;

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::processFaceBlur(state); // Algo Call

        cv::imshow(config.windowName, state.currentFrame);
        DisplayManager::handleInput(state);
    }
    cv::destroyAllWindows();
}

void AppController::executeTrackerMode() {
    if (!setupMode("Mode: Object Tracker")) return;
    cv::setMouseCallback(config.windowName, DisplayManager::onMouse, &state);

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::processObjectTracking(state); // Algo Call

        DisplayManager::render(state, config);
        DisplayManager::handleInput(state);
        
        if (cv::waitKey(1) == 'r') { // Quick reset logic
            state.isTracking = false;
            state.useRoi = false;
            if(state.tracker) state.tracker.release();
        }
    }
    cv::destroyAllWindows();
}

void AppController::executeCanvasMode() {
    if (!setupMode("Mode: Air Canvas")) return;

    while (state.isRunning) {
        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;
        cv::flip(state.currentFrame, state.currentFrame, 1);

        ImageProcessor::processAirCanvas(state, config); // Algo Call

        cv::imshow(config.windowName, state.currentFrame);
        
        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;
        if (key == 'c') {
            state.canvasStrokes.clear();
            state.currentStroke.clear();
        }
    }
    cv::destroyAllWindows();
}