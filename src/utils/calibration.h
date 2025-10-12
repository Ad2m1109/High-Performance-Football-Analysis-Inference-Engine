#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class Calibration {
public:
    Calibration(const std::string& calibration_path);

    std::vector<std::pair<int, cv::Point2f>> transform(const std::vector<std::pair<int, cv::Point2f>>& tracks);

    std::pair<int, cv::Point2f> transform(const std::pair<int, cv::Point2f>& track);

private:
    cv::Mat homography_matrix_;
};

#endif // CALIBRATION_H
