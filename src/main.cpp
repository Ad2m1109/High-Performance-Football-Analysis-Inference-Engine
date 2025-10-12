#include <iostream>
#include <vector>
#include "cxxopts.hpp"
#include "utils/config.h"
#include "detection/player_tracker.h"
#include "detection/ball_tracker.h"
#include "detection/yolov8.h" // Include the new YOLOv8 header
#include "analytics/metrics.h"
#include "utils/calibration.h"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
    cxxopts::Options options("SportsAnalytics", "A tool for analyzing football match videos.");

    options.add_options()
        ("v,video", "Path to the input video file", cxxopts::value<std::string>())
        ("c,calib", "Path to the camera calibration YAML file", cxxopts::value<std::string>())
        ("m,model", "Path to the YOLOv8 ONNX model file", cxxopts::value<std::string>())
        ("o,output-dir", "Directory to save the output CSV files", cxxopts::value<std::string>()->default_value("."))
        ("conf", "Confidence threshold for detection", cxxopts::value<float>()->default_value("0.5"))
        ("no-ball", "Disable ball tracking", cxxopts::value<bool>()->default_value("false"))
        ("skip-frames", "Number of frames to skip between analyses (e.g., 3 for every 3rd frame)", cxxopts::value<int>()->default_value("1"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("h")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    Config config;
    try {
        config.video_path = result["video"].as<std::string>();
        config.calibration_path = result["calib"].as<std::string>();
        config.yolo_model_path = result["model"].as<std::string>();
        config.output_dir = result["output-dir"].as<std::string>();
        config.confidence_threshold = result["conf"].as<float>();
        config.track_ball = !result["no-ball"].as<bool>();
        config.frame_skip_interval = result["skip-frames"].as<int>();
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return 1;
    }

    // Load calibration
    Calibration calibration(config.calibration_path);

    // Initialize YOLOv8 detector
    YoloV8 yolo_detector(config.yolo_model_path);

    // Initialize trackers
    PlayerTracker player_tracker;
    BallTracker ball_tracker;

    // Initialize metrics calculator
    MetricsCalculator metrics_calculator(config.output_dir);

    // Open video
    cv::VideoCapture cap(config.video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file " << config.video_path << std::endl;
        return 1;
    }

    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps == 0) {
        std::cerr << "Warning: Could not retrieve video FPS. Assuming 30 FPS." << std::endl;
        video_fps = 30.0; // Default to 30 FPS if not available
    }

    cv::Mat frame;
    int current_frame_idx = 0; // Actual frame index from video
    while (cap.read(frame)) {
        current_frame_idx++;

        // Skip frames if interval is greater than 1
        if (config.frame_skip_interval > 1 && (current_frame_idx - 1) % config.frame_skip_interval != 0) {
            continue; // Skip this frame
        }

        // Perform detection for all objects
        auto all_detections = yolo_detector.detect(frame);

        // Filter detections for players and the ball
        // COCO class IDs: 0 for person, 32 for sports ball
        std::vector<Detection> player_detections;
        std::vector<Detection> ball_detections;
        for (const auto& det : all_detections) {
            if (det.class_id == 0 && det.confidence >= config.confidence_threshold) {
                player_detections.push_back(det);
            } else if (det.class_id == 32 && det.confidence >= config.confidence_threshold) {
                ball_detections.push_back(det);
            }
        }

        // Update trackers with the new detections
        player_tracker.update(player_detections, frame); // Pass frame for color extraction

        if (config.track_ball) {
            ball_tracker.update(ball_detections);
        }

        // Convert to real-world coordinates
        auto real_world_players = calibration.transform(player_tracker.get_tracks());
        auto real_world_ball = calibration.transform(ball_tracker.get_track());

        // Calculate metrics
        // Team assignments are done after the loop, so pass an empty map for now
        // This is a simplified approach. A more robust solution would involve
        // storing all raw data and processing it once at the end.
        metrics_calculator.process_frame(current_frame_idx, video_fps, real_world_players, real_world_ball, player_tracker.get_team_assignments());
    }

    // Assign teams after all frames are processed (this is called again for final assignments)
    // The team assignments are already passed to process_frame, but this ensures final consistency
    player_tracker.assign_teams(); 

    // Save metrics to CSV
    metrics_calculator.save_to_csv();

    return 0;
}