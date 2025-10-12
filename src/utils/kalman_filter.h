#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <opencv2/opencv.hpp>

class KalmanFilter {
public:
    KalmanFilter();

    // Initialize the filter with an initial measurement
    void init(const cv::Point2f& measurement);

    // Predict the next state
    cv::Point2f predict();

    // Correct the state with a new measurement
    cv::Point2f correct(const cv::Point2f& measurement);

    // Get the current state (position) from the Kalman filter
    cv::Point2f get_state() const;

private:
    cv::KalmanFilter kf_;
    cv::Mat state_;       // [x, y, vx, vy]
    cv::Mat measurement_; // [x, y]
};

#endif // KALMAN_FILTER_H
