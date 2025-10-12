#ifndef CONFIG_H
#define CONFIG_H

#include <string>

struct Config {
    std::string video_path;
    std::string calibration_path;
    std::string output_dir;
    std::string yolo_model_path; // Consolidated model path
    float confidence_threshold;
    bool track_ball;
    int frame_skip_interval; // New member for frame skipping
};

#endif // CONFIG_H
