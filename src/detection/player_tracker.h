#ifndef PLAYER_TRACKER_H
#define PLAYER_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "detection/yolov8.h"
#include "utils/kalman_filter.h"
#include <map>
#include <string>

struct Track {
    int id;
    KalmanFilter kf;
    cv::Rect2f last_bbox;
    int frames_since_update = 0;
    cv::Scalar dominant_color; // Store dominant color (HSV)
};

class PlayerTracker {
public:
    PlayerTracker();
    ~PlayerTracker();

    void update(const std::vector<Detection>& detections, const cv::Mat& frame);

    void assign_teams();

    std::vector<std::pair<int, cv::Point2f>> get_tracks();

    const std::map<int, std::string>& get_team_assignments() const { return team_assignments_; }

private:
    int next_track_id_ = 0;
    std::vector<Track> tracks_;
    std::map<int, std::string> team_assignments_;
    const int max_frames_to_skip_ = 5;

    double calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2);
    cv::Scalar get_dominant_color(const cv::Mat& image_roi);
};

#endif // PLAYER_TRACKER_H