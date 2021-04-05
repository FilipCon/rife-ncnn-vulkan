#include "FrameInterpolation.h"
#include "ImageIO.h"
#include "Settings.h"
#include <chrono>
#include <iostream>

// start measuring time
#define START_CHRONO()                                                         \
    std::chrono::high_resolution_clock::time_point t1;                         \
    t1 = std::chrono::high_resolution_clock::now();

// end measuring time
#define END_CHRONO()                                                           \
    std::chrono::high_resolution_clock::time_point t2;                         \
    t2 = std::chrono::high_resolution_clock::now();                            \
    std::cout << "Elapsed time: "                                              \
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -    \
                                                                       t1)     \
                         .count()                                              \
              << "ms" << std::endl;

using namespace FrameInterpolation;

int main(int argc, char* argv[]) {
    // load images
    ncnn::Mat image0;
    ncnn::Mat image1;
    loadImage(std::string(DATA_DIR + "/I2_0.png"), image0);
    loadImage(std::string(DATA_DIR + "/I2_1.png"), image1);

    // select ncnn model
    std::string model = MODELS_DIR + "/rife-HD";

    // interpolate images
    Interpolation interpolator(model, 0, 0);
    interpolator.load(image0, image1);

    auto outImage = ncnn::Mat();
    interpolator.save(outImage);

    // save to file
    saveImage(DATA_DIR + "/I2_X.png", outImage);
    return 0;
}
