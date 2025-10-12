#include "utils/kalman_filter.h"

KalmanFilter::KalmanFilter() {
    // State: [x, y, vx, vy]'
    kf_ = cv::KalmanFilter(4, 2, 0);

    // State transition matrix (F)
    // x_k = x_{k-1} + dt * vx_{k-1}
    // y_k = y_{k-1} + dt * vy_{k-1}
    // vx_k = vx_{k-1}
    // vy_k = vy_{k-1}
    float dt = 1.0f;
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Measurement matrix (H)
    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Process noise covariance (Q)
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-1));

    // Measurement noise covariance (R)
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-2));

    // Error covariance post (P)
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0));

    state_ = cv::Mat(4, 1, CV_32F);
    measurement_ = cv::Mat(2, 1, CV_32F);
}

void KalmanFilter::init(const cv::Point2f& measurement) {
    kf_.statePost.at<float>(0) = measurement.x;
    kf_.statePost.at<float>(1) = measurement.y;
    kf_.statePost.at<float>(2) = 0;
    kf_.statePost.at<float>(3) = 0;
}

cv::Point2f KalmanFilter::predict() {
    cv::Mat prediction = kf_.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

cv::Point2f KalmanFilter::correct(const cv::Point2f& measurement) {
    measurement_.at<float>(0) = measurement.x;
    measurement_.at<float>(1) = measurement.y;
    cv::Mat corrected = kf_.correct(measurement_);
    return cv::Point2f(corrected.at<float>(0), corrected.at<float>(1));
}

cv::Point2f KalmanFilter::get_state() const {
    return cv::Point2f(kf_.statePost.at<float>(0), kf_.statePost.at<float>(1));
}
