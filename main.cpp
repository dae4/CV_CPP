#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    cout << "=== Start Camera Test ===" << endl;

    // 1. DirectShow(CAP_DSHOW)를 강제로 지정 (Media Foundation 지원 안 함)
    // 0번이 안 되면 1번으로 바꿔보세요.
    VideoCapture cap(0, CAP_DSHOW); 
     cout << "!!! Connecting Camera !!!" << endl;
    if (!cap.isOpened()) {
        cout << "!!! Camera connection failed !!!" << endl;
        return -1;
    }

    // 2. ★핵심★ 카메라 포맷과 해상도 강제 설정
    // 많은 USB 웹캠이 이 설정이 없으면 DirectShow에서 검은 화면이 나옵니다.
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // 설정이 잘 먹혔는지 확인
    double width = cap.get(CAP_PROP_FRAME_WIDTH);
    double height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Camera resolution: " << width << " x " << height << endl;

    cout << "Camera On. (End: Press ESE)" << endl;

    Mat frame, edge;
    while (true) {
        cap >> frame; 

        if (frame.empty()) {
            cout << "!!! Frame transmission failure (blank screen) !!!" << endl;
            // 카메라가 켜지는 중일 수 있으므로 잠깐 기다림
            waitKey(100);
            continue; 
        }
        Canny(frame, edge, 50, 150);

        imshow("Webcam Test", frame);
        imshow("Edge Detection", edge);
        if (waitKey(30) == 27) { // ESC 키
            break; 
        }
    }

    
    cap.release();
    destroyAllWindows();
    return 0;

}