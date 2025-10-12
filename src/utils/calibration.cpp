#include "utils/calibration.h"
#include "yaml-cpp/yaml.h"

Calibration::Calibration(const std::string& calibration_path) {
    try {
        YAML::Node config = YAML::LoadFile(calibration_path);
        const YAML::Node& h_matrix = config["homography_matrix"];
        homography_matrix_ = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                homography_matrix_.at<double>(i, j) = h_matrix[i * 3 + j].as<double>();
            }
        }
    } catch (const YAML::Exception& e) {
        // Handle exception
    }
}

std::vector<std::pair<int, cv::Point2f>> Calibration::transform(const std::vector<std::pair<int, cv::Point2f>>& tracks) {
    std::vector<std::pair<int, cv::Point2f>> transformed_tracks;
    for (const auto& track : tracks) {
        transformed_tracks.push_back(transform(track));
    }
    return transformed_tracks;
}

std::pair<int, cv::Point2f> Calibration::transform(const std::pair<int, cv::Point2f>& track) {
    cv::Point2f transformed_point;
    if (!homography_matrix_.empty()) {
        std::vector<cv::Point2f> src = {track.second};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, homography_matrix_);
        transformed_point = dst[0];
    }
    return {track.first, transformed_point};
}
