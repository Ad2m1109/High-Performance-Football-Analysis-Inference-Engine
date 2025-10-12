#include "detection/ball_tracker.h"

BallTracker::BallTracker() : track_({-1, {}}) {}

BallTracker::~BallTracker() {}

void BallTracker::update(const std::vector<Detection>& detections) {
    float max_confidence = 0.0f;
    const Detection* best_ball = nullptr;

    // Find the ball detection with the highest confidence
    for (const auto& det : detections) {
        if (det.confidence > max_confidence) {
            max_confidence = det.confidence;
            best_ball = &det;
        }
    }

    if (best_ball) {
        cv::Point2f center(best_ball->box.x + best_ball->box.width / 2.0f, 
                           best_ball->box.y + best_ball->box.height);
        if (!is_tracking_) {
            kf_.init(center);
            is_tracking_ = true;
        } else {
            kf_.correct(center);
        }
        frames_since_detection_ = 0;
    } else {
        if (is_tracking_) {
            frames_since_detection_++;
            if (frames_since_detection_ > max_frames_to_skip_) {
                is_tracking_ = false;
                track_ = {-1, {}};
            } else {
                kf_.predict();
            }
        }
    }

    if (is_tracking_) {
        cv::Point2f predicted_pos = kf_.predict();
        track_ = {0, predicted_pos}; // Use a fixed ID for the ball
    }
}

std::pair<int, cv::Point2f> BallTracker::get_track() {
    return track_;
}