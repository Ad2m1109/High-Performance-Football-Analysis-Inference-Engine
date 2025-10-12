#ifndef METRICS_H
#define METRICS_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

class MetricsCalculator {
public:
    MetricsCalculator(const std::string& output_dir);

    void process_frame(int frame_count, double fps, const std::vector<std::pair<int, cv::Point2f>>& player_tracks, const std::pair<int, cv::Point2f>& ball_track, const std::map<int, std::string>& team_assignments);

    void save_to_csv();

private:
    std::string output_dir_;
    std::vector<std::map<std::string, std::string>> player_metrics_;
    std::vector<std::map<std::string, std::string>> ball_metrics_;
    std::map<int, cv::Point2f> last_player_positions_; // For speed/distance calculation
    std::map<int, double> player_total_distances_; // For cumulative distance
    std::map<int, int> last_player_frame_counts_; // For accurate speed calculation with frame skipping
    std::map<int, int> player_frame_counts_; // number of frames seen per player
    double video_fps_ = 30.0; // default fps, updated from process_frame
};

#endif // METRICS_H
