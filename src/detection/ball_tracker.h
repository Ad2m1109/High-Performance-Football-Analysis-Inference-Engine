#ifndef BALL_TRACKER_H
#define BALL_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "detection/yolov8.h"
#include "utils/kalman_filter.h"

class BallTracker {
public:
    BallTracker();
    ~BallTracker();

    void update(const std::vector<Detection>& detections);

    std::pair<int, cv::Point2f> get_track();

private:
    KalmanFilter kf_;
    bool is_tracking_ = false;
    int frames_since_detection_ = 0;
    const int max_frames_to_skip_ = 10; // Allow more frames for ball occlusion
    std::pair<int, cv::Point2f> track_;
};

#endif // BALL_TRACKER_H
