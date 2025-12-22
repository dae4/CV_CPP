#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <string>
#include <opencv2/dnn.hpp>

//configuration parameters
struct AppConfig {
    int deviceID = 0;   //Camera device ID
    int width = 640;    
    int height = 480;

    
    // Motion detection parameters
    int thresholdVal = 30; 
    int minArea = 500;  // region area threshold
    std::string windowName = "Cv Application";

    //Air cnavas settings
    cv:: Scalar colorLower = cv:: Scalar(100, 150, 50);
    cv:: Scalar colorUpper = cv:: Scalar(140, 255, 255);

    // YOLO settings
    float yoloConfThreshold = 0.5f;
    float yoloNMSThreshold = 0.4f;
    float yoloScoreThreshold = 0.3f;
    std::string yoloModelPath = "yolov8n.onnx";
    std::string yoloClassPath = "classes.txt";
    
    // YOLO Segmentation settings
    std::string yoloSegModelPath = "C:\\Users\\3210m\\Desktop\\CV_CPP\\build\\yolov8n-seg.onnx";
    std::string yoloSegClassPath = "classes.txt";
    float yoloSegConfThreshold = 0.5f;
    float yoloSegNMSThreshold = 0.4f; 
    float yoloSegScoreThreshold = 0.3f;

};


// Runtime state variables
struct RuntimeState {
    cv::Mat currentFrame;  
    cv::Mat grayFrame;  
    cv::Mat prevFrame;
    cv::Mat diffFrame; 
    
    bool isRunning = true;     // Control main loop
    
    // Recording variables
    cv::VideoWriter writer;
    bool isRecording = false; 

     // Detected motion regions
    std::vector<cv::Rect> motionRects;
    int noMotionFrameCount = 0;

    // ROI selection variables
    cv::Rect roiRect;
    bool useRoi = false;
    bool isDragging = false;
    cv::Point startPoint;
    cv::Point curPoint;  
   
    // Face Blur
    bool faceModelLoaded = false;
    cv::CascadeClassifier faceCascade;
    cv::dnn::Net faceNet; 

    // Object Tracker
    bool isTracking = false;
    cv::Ptr<cv:: Tracker> tracker;
    

    // Air Canvas
    std::vector<std::vector<cv::Point>> canvasStrokes;
    std::vector<cv::Point> currentStroke;

    // YOLO Object Detection
    cv::dnn::Net yoloNet;
    std::vector<std::string> yoloClasses;
    bool yoloModelLoaded = false;

    // Traffic Counting
    int trafficCount = 0;
    std::map<int, cv::Point> prevPoints;
    int nextObjectID = 0;

    // YOLO Instance Segmentation
    cv::dnn::Net yoloSegNet;
    std::vector<std::string> yoloSegClasses;
    bool yoloSegModelLoaded = false;
};

