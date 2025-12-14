#define _CRT_SECURE_NO_WARNINGS 
#include "ImageProcessor.hpp"
#include <iomanip> 
#include <ctime>   
#include <sstream> 
#include <iostream>

namespace ImageProcessor { 

    // Helper: Generate timestamp string for filenames
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
        cv::GaussianBlur(dst, dst, cv::Size(21, 21), 0); // Reduce noise
    }

    // --- [Feature 1] Motion Detector Implementation ---
    void detectMotion(RuntimeState& state, const AppConfig& config) {
        cv::Mat processArea;

        // Determine Processing Area (ROI vs Full Frame)
        if (state.useRoi) {
            cv::Rect safeRoi = state.roiRect & cv::Rect(0, 0, state.grayFrame.cols, state.grayFrame.rows);
            if (safeRoi.area() > 0)
                processArea = state.grayFrame(safeRoi);
            else 
                processArea = state.grayFrame;
        } else {
            processArea = state.grayFrame;
        }

        // Initialize background model (first frame)
        if (state.prevFrame.empty() || state.prevFrame.size() != processArea.size()) {
           state.prevFrame = processArea.clone();
           return;
        }

        // 1. Calculate Difference (Current - Previous)
        cv::absdiff(state.prevFrame, processArea, state.diffFrame);
        
        // 2. Thresholding & Dilation (Morphology)
        cv::threshold(state.diffFrame, state.diffFrame, config.thresholdVal, 255, cv::THRESH_BINARY); 
        cv::dilate(state.diffFrame, state.diffFrame, cv::Mat(), cv::Point(-1, -1), 2);

        // 3. Find Contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(state.diffFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        state.motionRects.clear();
        for (const auto& contour : contours) { 
            if (cv::contourArea(contour) < config.minArea) continue; // Filter small noise
            
            cv::Rect r = cv::boundingRect(contour);

            // Adjust coordinates if ROI is active
            if (state.useRoi) {
                r.x += state.roiRect.x; 
                r.y += state.roiRect.y; 
            }
            state.motionRects.push_back(r);
        }

        // 4. Handle Recording Logic
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

        // Update previous frame
        state.prevFrame = processArea.clone();
    }

    // --- [Feature 2] Optical Flow Implementation ---
    void computeOpticalFlow(RuntimeState& state) {
        cv::Mat currentGray;
        cv::cvtColor(state.currentFrame, currentGray, cv::COLOR_BGR2GRAY);

        if (state.prevFrame.empty() || state.prevFrame.size() != currentGray.size()) {
            state.prevFrame = currentGray.clone();
            return;
        }

        cv::Mat flow;
        // Calculate Dense Optical Flow using Farneback algorithm
        cv::calcOpticalFlowFarneback(state.prevFrame, currentGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // Draw Flow Vectors (Grid-based)
        for (int y = 0; y < flow.rows; y += 10) {
            for (int x = 0; x < flow.cols; x += 10) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                if (fxy.x * fxy.x + fxy.y * fxy.y > 1) { 
                    cv::line(state.currentFrame, cv::Point(x, y), 
                             cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 
                             cv::Scalar(0, 255, 0));
                    cv::circle(state.currentFrame, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
        state.prevFrame = currentGray.clone();
    }

    // --- [Feature 3] Face Blur Implementation ---
    // using DNN-based face detector for better accuracy
    void processFaceBlur(RuntimeState& state) {
        // 1. Load Model (Lazy Loading)
        if (!state.faceModelLoaded) {
            try {
                // Load the Caffe model (Ensure these files are in your project directory)
                state.faceNet = cv::dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
                
                // Optional: Use GPU if available
                state.faceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                state.faceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                
                state.faceModelLoaded = true;
                std::cout << "[Info] DNN Face Model Loaded Successfully." << std::endl;
            } catch (cv::Exception& e) {
                std::cerr << "[Error] Failed to load DNN Net: " << e.what() << std::endl;
                std::cerr << ">> Please download 'deploy.prototxt' and '.caffemodel'!" << std::endl;
                return;
            }
        }

        // 2. Create a 4D blob from the frame (Resize to 300x300 for the model)
        // Mean subtraction values: (104.0, 177.0, 123.0)
        cv::Mat blob = cv::dnn::blobFromImage(state.currentFrame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
        state.faceNet.setInput(blob);

        // 3. Forward pass (Inference)
        cv::Mat detection = state.faceNet.forward();

        // 4. Parse the result
        // Detection matrix structure: [batch, class, confidence, x1, y1, x2, y2]
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            // Filter out weak detections (Confidence threshold: 50%)
            if (confidence > 0.5) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * state.currentFrame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * state.currentFrame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * state.currentFrame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * state.currentFrame.rows);

                // Create rectangle and clip to frame boundaries
                cv::Rect faceRect(cv::Point(x1, y1), cv::Point(x2, y2));
                faceRect = faceRect & cv::Rect(0, 0, state.currentFrame.cols, state.currentFrame.rows);

                if (faceRect.area() > 0) {
                    // Apply strong Gaussian Blur to the face region
                    cv::Mat faceROI = state.currentFrame(faceRect);
                    cv::GaussianBlur(faceROI, faceROI, cv::Size(0, 0), 30); 
                    
                    // Draw bounding box
                    cv::rectangle(state.currentFrame, faceRect, cv::Scalar(0, 255, 0), 2);
                    
                    // Display confidence score
                    std::string label = cv::format("Face: %.2f", confidence);
                    cv::putText(state.currentFrame, label, cv::Point(x1, y1 - 5), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
                }
            }
        }
    }

    // --- [Feature 4] Object Tracker Implementation ---
    // Using CSRT Tracker for better accuracy and occlusion handling

    void processObjectTracking(RuntimeState& state) {
        // Initialize Tracker if ROI is selected
        if (state.useRoi && !state.isTracking) {
            // [CHANGE] Use CSRT instead of KCF for better accuracy and occlusion handling
            state.tracker = cv::TrackerCSRT::create(); 
            state.tracker->init(state.currentFrame, state.roiRect);
            state.isTracking = true;
            state.useRoi = false;
            std::cout << "[Info] CSRT Tracker Initialized." << std::endl;
        }

        if (state.isTracking) {
            bool ok = state.tracker->update(state.currentFrame, state.roiRect);
            if (ok) {
                // Tracking success
                cv::rectangle(state.currentFrame, state.roiRect, cv::Scalar(255, 0, 0), 2, 1);
                cv::putText(state.currentFrame, "Tracking (CSRT)", cv::Point(20, 50), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
            } else {
                // Tracking failure
                cv::putText(state.currentFrame, "Object Lost", cv::Point(20, 50), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
            }
        }
    }
    // --- [Feature 5] Air Canvas Implementation ---
    void processAirCanvas(RuntimeState& state, const AppConfig& config) {
        cv::Mat hsv, mask;
        
        // 1. Pre-process: Blur the frame to reduce high-frequency noise
        cv::GaussianBlur(state.currentFrame, hsv, cv::Size(11, 11), 0);
        cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);

        // 2. Color Thresholding
        cv::inRange(hsv, config.colorLower, config.colorUpper, mask);

        // 3. Morphological Operations (Crucial for clean lines)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        
        // Opening (Erosion -> Dilation): Removes small dots/noise
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel); 
        
        // Closing (Dilation -> Erosion): Fills small holes inside the object
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // 4. Contour Detection
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Point center;
        bool objectFound = false;

        if (!contours.empty()) {
            // Find the largest contour (assumed to be the drawing tool)
            auto maxContour = std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            // [CHANGE] Increased area threshold to ignore background noise
            if (cv::contourArea(*maxContour) > 1000) { 
                cv::Moments M = cv::moments(*maxContour);
                center = cv::Point(int(M.m10 / M.m00), int(M.m01 / M.m00));
                objectFound = true;

                // Draw pointer tip
                cv::circle(state.currentFrame, center, 10, cv::Scalar(0, 255, 255), 2); 
                state.currentStroke.push_back(center);
            }
        }

        // 5. Handle Stroke Logic
        // If object is lost or not found, finalize the current stroke
        if (!objectFound && !state.currentStroke.empty()) {
            state.canvasStrokes.push_back(state.currentStroke);
            state.currentStroke.clear();
        }

        // 6. Draw Canvas
        // Draw historical strokes
        for (const auto& stroke : state.canvasStrokes) {
            if (stroke.size() < 2) continue;
            for (size_t i = 1; i < stroke.size(); i++) {
                cv::line(state.currentFrame, stroke[i - 1], stroke[i], cv::Scalar(255, 255, 0), 3);
            }
        }
        // Draw active stroke
        for (size_t i = 1; i < state.currentStroke.size(); i++) {
            cv::line(state.currentFrame, state.currentStroke[i - 1], state.currentStroke[i], cv::Scalar(255, 255, 0), 3);
        }
    }
}