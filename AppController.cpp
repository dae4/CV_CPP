#include "AppController.hpp"
#include "Timer.hpp"    

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
        std::cout << " 6. YOLO Object Detection" << std::endl;
        std::cout << " 7. Traffic Counting" << std::endl;
        std::cout << " 8. YOLO Instance Segmentation" << std::endl; 
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
            case 6: executeYOLOMode(); break;
            case 7: executeTrafficMode(); break;
            case 8: executeYOLOSegmentationMode(); break;
            default: std::cout << "Invalid Selection." << std::endl;
        }
    }
}

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

void AppController::executeYOLOMode() {
    if (!setupMode("Mode: YOLO Object Detection")) return;
    
    UtilityTimer timer;

    while (state.isRunning) {
        timer.update();

        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::processYOLODetection(state, config); // Algo Call

        ImageProcessor::renderFPS(state.currentFrame, timer.getFpsString()); // Render FPS

        cv::imshow(config.windowName, state.currentFrame);
        DisplayManager::handleInput(state);
    }
    cv::destroyAllWindows();
}

void AppController::executeTrafficMode() {
    // 준비한 영상 경로로 수정하세요
    std::string videoPath = "../asset/cctv.mp4"; 
    cap.open(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file!" << std::endl;
        return;
    }

    UtilityTimer timer;
    while (state.isRunning) {
        timer.update();
        cv::Mat rawFrame;
        cap >> rawFrame;
        if (rawFrame.empty()) break; // 영상 끝나면 종료

        cv::resize(rawFrame, state.currentFrame, cv::Size(640, 480), 0, 0, cv::INTER_AREA);
        
        ImageProcessor::processTrafficCounting(state, config);
        ImageProcessor::renderFPS(state.currentFrame, timer.getFpsString());

        cv::imshow(config.windowName, state.currentFrame);
        if (DisplayManager::handleInput(state) == 'q') break;
    }
    cap.release();
}

void AppController::executeYOLOSegmentationMode() {
    if (!setupMode("Mode: YOLO Instance Segmentation")) return;
    
    UtilityTimer timer;

    while (state.isRunning) {
        timer.update();

        cap >> state.currentFrame;
        if (state.currentFrame.empty()) break;

        ImageProcessor::processYOLOSegmentation(state, config); // Algo Call

        ImageProcessor::renderFPS(state.currentFrame, timer.getFpsString()); // Render FPS

        cv::imshow(config.windowName, state.currentFrame);
        DisplayManager::handleInput(state);
    }
    cv::destroyAllWindows();
}