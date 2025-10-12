#ifndef YOLOV8_H
#define YOLOV8_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

// Struct to hold detection results
struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class YoloV8 {
public:
    // Constructor: Takes the path to the ONNX model file
    YoloV8(const std::string& onnx_model_path);

    // Destructor
    ~YoloV8();

    // Main detection function
    std::vector<Detection> detect(const cv::Mat& image);

private:
    // --- TensorRT Members ---
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    
    // --- Model Info ---
    std::string onnx_model_path_;
    std::string engine_file_path_;
    const int input_width_ = 640;
    const int input_height_ = 640;

    // --- Buffers ---
    void* buffers_[2]; // 0 for input, 1 for output
    cudaStream_t stream_ = nullptr;

    // --- Initialization ---
    void buildEngine();
    void loadEngine();

    // --- Inference Helpers ---
    std::vector<float> preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const float* output, const cv::Size& original_image_size);
};

#endif // YOLOV8_H
