#include "FrameInterpolation.h"
#include "ImageIO.h"
#include "Measure.h"
#include "Settings.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    // load video
    auto videoName = std::string(DATA_DIR + "/Las-Vegas-short.webm");
    cv::VideoCapture capture(videoName);
    cv::Mat frame, prevFrame;
    capture >> frame;

    int w = frame.cols;
    int h = frame.rows;

    // set final images
    cv::Mat result(h, w, frame.type());

    // select ncnn model
    std::string model = MODELS_DIR + "/rife-v2.4";

    // interpolate images
    Interpolation interpolator(model, false, true);

    // set number of interpolation
    int totalTimeSteps = 2;

    auto in0imageMat = ncnn::Mat(w, h, prevFrame.data, 3, 3);
    auto in1imageMat = ncnn::Mat(w, h, frame.data, 3, 3);
    auto outimageMat = ncnn::Mat(w, h, result.data, 3, 3);

    while (true) {
        capture >> frame;
        if (!frame.data) break;

        if (prevFrame.data) {
            for (int t = 0; t < totalTimeSteps; ++t) {
                double timeStep = 1.0 / totalTimeSteps * t;

                START_CHRONO();
                interpolator.interpolate(prevFrame.data, frame.data,
                                         result.data, frame.cols, frame.rows);
                END_CHRONO();

                result.data = (uchar*) outimageMat.data;

                imshow("Test", result);
                cv::waitKey(1);
            }
        }
        swap(frame, prevFrame);
    }

    return 0;
}
