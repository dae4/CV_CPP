#define _CRT_SECURE_NO_WARNINGS 
#include "ImageProcessor.hpp"
#include <iomanip> 
#include <ctime>   
#include <sstream> 
#include <iostream>
#include <fstream>

namespace ImageProcessor { 

    // Utility: Render FPS on frame
    void renderFPS(cv::Mat& frame, const std::string& fpsString) {
        cv::putText(frame, fpsString, cv::Point(frame.cols - 180, 30), // Top Right Corner
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2); // Yellow color
    }

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

    // --- [Feature 6] YOLO Object Detection Implementation ---
    void processYOLODetection(RuntimeState& state, const AppConfig& config) {
        
        if (!state.yoloModelLoaded) {
            try {
                // 클래스 이름 파일 읽기
                std::ifstream ifs(config.yoloClassPath);
                if (!ifs.is_open()) {
                    std::cerr << "[Error] Failed to find 'classes.txt'!" << std::endl;
                    return;
                }
                std::string line;
                while (std::getline(ifs, line)) state.yoloClasses.push_back(line);

                // YOLO 모델 로드 (ONNX)
                state.yoloNet = cv::dnn::readNetFromONNX(config.yoloModelPath);
                
                // GPU 사용 가능 시 가속 (없으면 자동 CPU)
                state.yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                state.yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

                state.yoloModelLoaded = true;
                std::cout << "[Info] YOLOv8 Model Loaded. Classes: " << state.yoloClasses.size() << std::endl;

            } catch (cv::Exception& e) {
                std::cerr << "[Error] YOLO Init: " << e.what() << std::endl;
                std::cerr << ">> Check 'yolov8n.onnx' and 'classes.txt'!" << std::endl;
                return;
            }
        }

        if (state.currentFrame.empty()) {
            std::cerr << "[Critical Error] Input Frame is EMPTY inside processYOLO!" << std::endl;
            return;
        }
        std::cout << "[Debug] Frame Size: " << state.currentFrame.cols << " x " << state.currentFrame.rows << std::endl;
        

        // 2. 전처리 (Blob 생성)
        // YOLOv8은 640x640 입력, 0~1 사이 값(1/255 scale), RGB 포맷을 원함
        cv::Mat blob = cv::dnn::blobFromImage(state.currentFrame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        
        if (blob.empty()) {
            std::cerr << "[Error] Blob is empty!" << std::endl;
            return;
        }
        std::cout << "[Debug] Blob size: " << blob.size << std::endl; // [1, 3, 640, 640]이 나와야 함
        state.yoloNet.setInput(blob);

        // 3. 추론 (Inference)
        std::vector<cv::Mat> outputs;
        state.yoloNet.forward(outputs, state.yoloNet.getUnconnectedOutLayersNames());

        // 4. 후처리 (Post-processing)
        // YOLOv8 Output shape: [1, 84, 8400] -> (cx, cy, w, h, 80개 클래스 점수)
        int dimensions = outputs[0].size[1];
        int rows = outputs[0].size[2];
        
        // 데이터 파싱을 위해 차원 변환: [1, 84, 8400] -> [8400, 84] 로 전치(Transpose)
        cv::Mat output_buffer = outputs[0].reshape(1, 84); 
        cv::Mat output = output_buffer.t(); 
        
        if (dimensions > rows) { 
            // [1, 8400, 84] 형태라면 그대로 사용
            output = outputs[0].reshape(1, dimensions);
        } else {
            // [1, 84, 8400] 형태라면 전치(Transpose) 필요 (기존 코드)
            cv::Mat output_buffer = outputs[0].reshape(1, dimensions);
            output = output_buffer.t();
        }

        float* data = (float*)output.data;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = (float)state.currentFrame.cols / 640.0f;
        float y_factor = (float)state.currentFrame.rows / 640.0f;

        int detectCount = 0; // [디버깅] 몇 개나 찾았나 세어보자

        for (int i = 0; i < rows; ++i) {
            float* classes_scores = data + 4;
            cv::Mat scores(1, state.yoloClasses.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // [디버깅] 점수가 설정값보다 높으면 로그 출력
            if (max_class_score > config.yoloScoreThreshold) {
                
                // ★★★ 여기가 핵심! 찾았으면 로그를 찍어라 ★★★
                std::cout << "[YOLO] Found Class: " << class_id.x 
                          << " (" << (state.yoloClasses.empty() ? "?" : state.yoloClasses[class_id.x]) << ")"
                          << " Score: " << max_class_score << std::endl;

                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                if (i == 0) {
                   std::cout << "[DEBUG] Raw Coords: " << cx << ", " << cy << ", " << w << ", " << h << std::endl;
                }
                
                if (cx < 1.0f && cy < 1.0f && w < 1.0f && h < 1.0f) {
                    cx *= 640.0f;
                    cy *= 640.0f;
                    w *= 640.0f;
                    h *= 640.0f;
                }

                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back((float)max_class_score);
                class_ids.push_back(class_id.x);
                detectCount++;
            }
            data += 84;
        }

        // 5. NMS (Non-Maximum Suppression)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, config.yoloConfThreshold, config.yoloNMSThreshold, indices);

        // [디버깅] 최종적으로 남은 박스 개수
        if (detectCount > 0) {
            std::cout << "[YOLO] Raw Detections: " << detectCount << " -> After NMS: " << indices.size() << std::endl;
        }

        // 6. 그리기 (Drawing) - 이 부분이 없으면 화면에 안 나옵니다!
        for (int i : indices) {
            cv::Rect box = boxes[i];
            int classId = class_ids[i];
            float conf = confidences[i];

            // 박스 그리기
            cv::rectangle(state.currentFrame, box, cv::Scalar(0, 255, 0), 2);

            // 텍스트 라벨
            std::string className = (classId < state.yoloClasses.size()) ? state.yoloClasses[classId] : "Unknown";
            std::string label = className + " " + cv::format("%.2f", conf);
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            // 라벨 배경 (검은색)
            cv::rectangle(state.currentFrame, 
                          cv::Point(box.x, box.y - labelSize.height - 5), 
                          cv::Point(box.x + labelSize.width, box.y), 
                          cv::Scalar(255, 255, 255), -1);
            // 라벨 글씨 (흰색)
            cv::putText(state.currentFrame, label, cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

    }


    void processTrafficCounting(RuntimeState& state, const AppConfig& config) {
        if (!state.yoloModelLoaded) {
            try {
                // 클래스 이름 파일 읽기
                std::ifstream ifs(config.yoloClassPath);
                if (!ifs.is_open()) {
                    std::cerr << "[Error] Failed to find 'classes.txt'!" << std::endl;
                    return;
                }
                std::string line;
                while (std::getline(ifs, line)) state.yoloClasses.push_back(line);

                // YOLO 모델 로드 (ONNX)
                state.yoloNet = cv::dnn::readNetFromONNX(config.yoloModelPath);
                
                // GPU 사용 가능 시 가속 (없으면 자동 CPU)
                state.yoloNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                state.yoloNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

                state.yoloModelLoaded = true;
                std::cout << "[Info] YOLOv8 Model Loaded. Classes: " << state.yoloClasses.size() << std::endl;

            } catch (cv::Exception& e) {
                std::cerr << "[Error] YOLO Init: " << e.what() << std::endl;
                std::cerr << ">> Check 'yolov8n.onnx' and 'classes.txt'!" << std::endl;
                return;
            }
        }

        if (state.currentFrame.empty()) {
            std::cerr << "[Critical Error] Input Frame is EMPTY inside processYOLO!" << std::endl;
            return;
        }
        std::cout << "[Debug] Frame Size: " << state.currentFrame.cols << " x " << state.currentFrame.rows << std::endl;
        

        // 2. 전처리 (Blob 생성)
        // YOLOv8은 640x640 입력, 0~1 사이 값(1/255 scale), RGB 포맷을 원함
        cv::Mat blob = cv::dnn::blobFromImage(state.currentFrame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        
        if (blob.empty()) {
            std::cerr << "[Error] Blob is empty!" << std::endl;
            return;
        }
        std::cout << "[Debug] Blob size: " << blob.size << std::endl; // [1, 3, 640, 640]이 나와야 함
        state.yoloNet.setInput(blob);

        // 3. 추론 (Inference)
        std::vector<cv::Mat> outputs;
        state.yoloNet.forward(outputs, state.yoloNet.getUnconnectedOutLayersNames());

        // 4. 후처리 (Post-processing)
        // YOLOv8 Output shape: [1, 84, 8400] -> (cx, cy, w, h, 80개 클래스 점수)
        int dimensions = outputs[0].size[1];
        int rows = outputs[0].size[2];
        
        // 데이터 파싱을 위해 차원 변환: [1, 84, 8400] -> [8400, 84] 로 전치(Transpose)
        cv::Mat output_buffer = outputs[0].reshape(1, 84); 
        cv::Mat output = output_buffer.t(); 
        
        if (dimensions > rows) { 
            // [1, 8400, 84] 형태라면 그대로 사용
            output = outputs[0].reshape(1, dimensions);
        } else {
            // [1, 84, 8400] 형태라면 전치(Transpose) 필요 (기존 코드)
            cv::Mat output_buffer = outputs[0].reshape(1, dimensions);
            output = output_buffer.t();
        }

        float* data = (float*)output.data;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = (float)state.currentFrame.cols / 640.0f;
        float y_factor = (float)state.currentFrame.rows / 640.0f;

        int detectCount = 0; // [디버깅] 몇 개나 찾았나 세어보자

        for (int i = 0; i < rows; ++i) {
            float* classes_scores = data + 4;
            cv::Mat scores(1, state.yoloClasses.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // [디버깅] 점수가 설정값보다 높으면 로그 출력
            if (max_class_score > config.yoloScoreThreshold) {
                
                // ★★★ 여기가 핵심! 찾았으면 로그를 찍어라 ★★★
                std::cout << "[YOLO] Found Class: " << class_id.x 
                          << " (" << (state.yoloClasses.empty() ? "?" : state.yoloClasses[class_id.x]) << ")"
                          << " Score: " << max_class_score << std::endl;

                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                if (i == 0) {
                   std::cout << "[DEBUG] Raw Coords: " << cx << ", " << cy << ", " << w << ", " << h << std::endl;
                }
                
                if (cx < 1.0f && cy < 1.0f && w < 1.0f && h < 1.0f) {
                    cx *= 640.0f;
                    cy *= 640.0f;
                    w *= 640.0f;
                    h *= 640.0f;
                }

                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back((float)max_class_score);
                class_ids.push_back(class_id.x);
                detectCount++;
            }
            data += 84;
        }

        // 5. NMS (Non-Maximum Suppression)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, config.yoloConfThreshold, config.yoloNMSThreshold, indices);

        int lineY = state.currentFrame.rows * 0.7;
        cv::line(state.currentFrame, cv::Point(0, lineY), cv::Point(state.currentFrame.cols, lineY), cv::Scalar(0, 0, 255), 3);

        std::map<int, cv::Point> currentPoints;

        for (int i : indices) {
            cv::Rect box = boxes[i];
            cv::Point center(box.x + box.width / 2, box.y + box.height / 2);

            // [간이 트래킹] 이전 프레임의 점들 중 가장 가까운 점 찾기
            int matchedID = -1;
            double minDist = 50.0; // 50픽셀 이내여야 같은 객체로 인정

            for (auto& prev : state.prevPoints) {
                double dist = cv::norm(center - prev.second);
                if (dist < minDist) {
                    minDist = dist;
                    matchedID = prev.first;
                }
            }

            if (matchedID == -1) {
                matchedID = state.nextObjectID++; // 새 객체 등록
            }

            // [카운팅 로직] 선을 위에서 아래로 통과했는가?
            if (state.prevPoints.count(matchedID)) {
                if (state.prevPoints[matchedID].y < lineY && center.y >= lineY) {
                    state.trafficCount++;
                    std::cout << "[Counter] Vehicle Detected! Total: " << state.trafficCount << std::endl;
                }
            }

            currentPoints[matchedID] = center;

            // 화면 표시
            cv::rectangle(state.currentFrame, box, cv::Scalar(0, 255, 0), 2);
            cv::circle(state.currentFrame, center, 4, cv::Scalar(255, 0, 0), -1);
            cv::putText(state.currentFrame, "ID:" + std::to_string(matchedID), cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        state.prevPoints = currentPoints; // 현재 점들을 다음 프레임용으로 저장

        // 카운트 결과 화면 출력
        cv::putText(state.currentFrame, "TOTAL COUNT: " + std::to_string(state.trafficCount), 
                    cv::Point(20, 50), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
    }
}
