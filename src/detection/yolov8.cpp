#include "yolov8.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "utils/logger.h" // Use the existing logger
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp> // For NMS

// Constructor
YoloV8::YoloV8(const std::string& onnx_model_path) : onnx_model_path_(onnx_model_path) {
    engine_file_path_ = onnx_model_path + ".engine";
    std::ifstream engine_file(engine_file_path_, std::ios::binary);

    if (engine_file.good()) {
        std::cout << "Loading existing TensorRT engine: " << engine_file_path_ << std::endl;
        loadEngine();
    } else {
        std::cout << "Building TensorRT engine from ONNX file: " << onnx_model_path_ << std::endl;
        buildEngine();
    }

    if (!engine_) {
        throw std::runtime_error("Failed to initialize TensorRT engine.");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }
    
    cudaStreamCreate(&stream_);

    // Allocate buffers
    cudaMalloc(&buffers_[0], input_width_ * input_height_ * 3 * sizeof(float));
    // The output tensor size is 8400 * (4+80) = 705600
    cudaMalloc(&buffers_[1], 8400 * 84 * sizeof(float));
}

// Destructor
YoloV8::~YoloV8() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
    delete context_;
    delete engine_;
    delete runtime_;
}

void YoloV8::buildEngine() {
    auto builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder.");
    }

    auto network = builder->createNetworkV2(0U);
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network.");
    }

    auto config = builder->createBuilderConfig();
    if (!config) {
        throw std::runtime_error("Failed to create TensorRT builder config.");
    }

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser.");
    }

    if (!parser->parseFromFile(onnx_model_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX file.");
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB

    nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *config);
    if (!serialized_engine) {
        throw std::runtime_error("Failed to build serialized network.");
    }

    std::ofstream engine_file(engine_file_path_, std::ios::binary);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size());

    delete serialized_engine;
    delete parser;
    delete config;
    delete network;
    delete builder;
}

void YoloV8::loadEngine() {
    std::ifstream engine_file(engine_file_path_, std::ios::binary);
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    engine_file.read(buffer.data(), size);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
}

std::vector<Detection> YoloV8::detect(const cv::Mat& image) {
    std::vector<float> preprocessed_image = preprocess(image);

    cudaMemcpyAsync(buffers_[0], preprocessed_image.data(), preprocessed_image.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);

    context_->setTensorAddress(engine_->getIOTensorName(0), buffers_[0]);
    context_->setTensorAddress(engine_->getIOTensorName(1), buffers_[1]);

    context_->enqueueV3(stream_);

    int output_size = 8400 * 84;
    std::vector<float> output_data(output_size);
    cudaMemcpyAsync(output_data.data(), buffers_[1], output_data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    return postprocess(output_data.data(), image.size());
}

std::vector<float> YoloV8::preprocess(const cv::Mat& image) {
    cv::Mat resized_image, float_image, rgb_image;
    
    cv::resize(image, resized_image, cv::Size(input_width_, input_height_));
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    
    std::vector<float> result(input_width_ * input_height_ * 3);
    memcpy(result.data(), channels[0].data, input_width_ * input_height_ * sizeof(float));
    memcpy(result.data() + input_width_ * input_height_, channels[1].data, input_width_ * input_height_ * sizeof(float));
    memcpy(result.data() + 2 * input_width_ * input_height_, channels[2].data, input_width_ * input_height_ * sizeof(float));

    return result;
}

std::vector<Detection> YoloV8::postprocess(const float* output, const cv::Size& original_image_size) {
    const int num_detections = 8400;
    const int num_classes = 80;
    const int elements_per_detection = num_classes + 4;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    std::vector<float> transposed_output(num_detections * elements_per_detection);
    for (int i = 0; i < num_detections; ++i) {
        for (int j = 0; j < elements_per_detection; ++j) {
            transposed_output[i * elements_per_detection + j] = output[j * num_detections + i];
        }
    }

    float scale_x = static_cast<float>(original_image_size.width) / input_width_;
    float scale_y = static_cast<float>(original_image_size.height) / input_height_;

    for (int i = 0; i < num_detections; ++i) {
        const float* detection = transposed_output.data() + i * elements_per_detection;
        const float* class_scores = detection + 4;
        
        int class_id = -1;
        float max_score = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            if (class_scores[j] > max_score) {
                max_score = class_scores[j];
                class_id = j;
            }
        }

        if (max_score > 0.25f) { 
            float cx = detection[0];
            float cy = detection[1];
            float w = detection[2];
            float h = detection[3];

            int left = static_cast<int>((cx - 0.5 * w) * scale_x);
            int top = static_cast<int>((cy - 0.5 * h) * scale_y);
            int width = static_cast<int>(w * scale_x);
            int height = static_cast<int>(h * scale_y);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.4f, 0.5f, nms_indices);

    std::vector<Detection> final_detections;
    for (int index : nms_indices) {
        final_detections.push_back({boxes[index], confidences[index], class_ids[index]});
    }

    return final_detections;
}
