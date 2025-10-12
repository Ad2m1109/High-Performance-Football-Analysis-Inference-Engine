#include "detection/player_tracker.h"
#include <algorithm> // For std::max
#include <numeric>   // For std::iota
#include <opencv2/imgproc.hpp> // For cvtColor, kmeans
#include <map> // For std::map
#include <set> // For std::set
#include <iostream> // For debugging, can be removed later

PlayerTracker::PlayerTracker() : next_track_id_(0) {}

PlayerTracker::~PlayerTracker() {}

double PlayerTracker::calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float box1_area = box1.width * box1.height;
    float box2_area = box2.width * box2.height;
    float union_area = box1_area + box2_area - intersection_area;

    return (union_area > 0) ? (intersection_area / union_area) : 0;
}

cv::Scalar PlayerTracker::get_dominant_color(const cv::Mat& image_roi) {
    if (image_roi.empty() || image_roi.rows == 0 || image_roi.cols == 0) {
        return cv::Scalar(0, 0, 0); // Return black for empty ROI
    }

    // Define a central region of interest (e.g., 25% to 75% of height and width)
    // to focus on the jersey and avoid skin/hair/background.
    int h = image_roi.rows;
    int w = image_roi.cols;
    cv::Rect jersey_region_rect(w * 0.25, h * 0.25, w * 0.5, h * 0.5);

    // Ensure the region is within bounds
    jersey_region_rect = jersey_region_rect & cv::Rect(0, 0, w, h);

    if (jersey_region_rect.empty()) {
        return cv::Scalar(0, 0, 0);
    }

    cv::Mat jersey_region = image_roi(jersey_region_rect);

    // Convert to HSV color space
    cv::Mat hsv_jersey;
    cv::cvtColor(jersey_region, hsv_jersey, cv::COLOR_BGR2HSV);

    // Reshape to a 1D array of pixels
    cv::Mat pixels = hsv_jersey.reshape(1, hsv_jersey.total());
    pixels.convertTo(pixels, CV_32F); // Convert to float for k-means

    // Apply k-means to find the dominant color (cluster with K=1)
    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);
    cv::kmeans(pixels, 1, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

    // The dominant color is the center of the single cluster
    cv::Scalar dominant_hsv(centers.at<float>(0, 0), centers.at<float>(0, 1), centers.at<float>(0, 2));

    return dominant_hsv;
}


void PlayerTracker::update(const std::vector<Detection>& detections, const cv::Mat& frame) {
    // 1. Predict new locations of existing tracks
    for (auto& track : tracks_) {
        cv::Point2f predicted_pos = track.kf.predict();
        track.last_bbox.x = predicted_pos.x - track.last_bbox.width / 2.0f;
        track.last_bbox.y = predicted_pos.y - track.last_bbox.height;
        track.frames_since_update++;
    }

    // 2. Associate detections with existing tracks using IOU
    std::vector<bool> matched_detections(detections.size(), false);
    std::vector<int> track_indices(tracks_.size());
    std::iota(track_indices.begin(), track_indices.end(), 0);

    for (int i = 0; i < tracks_.size(); ++i) {
        double max_iou = 0.0;
        int best_match_idx = -1;

        for (int j = 0; j < detections.size(); ++j) {
            if (!matched_detections[j]) {
                double iou = calculate_iou(tracks_[i].last_bbox, detections[j].box);
                if (iou > max_iou) {
                    max_iou = iou;
                    best_match_idx = j;
                }
            }
        }

        if (max_iou > 0.3) { // IOU threshold
            cv::Point2f detection_center(detections[best_match_idx].box.x + detections[best_match_idx].box.width / 2.0f, 
                                       detections[best_match_idx].box.y + detections[best_match_idx].box.height);
            tracks_[i].kf.correct(detection_center);
            tracks_[i].last_bbox = detections[best_match_idx].box;
            tracks_[i].frames_since_update = 0;
            matched_detections[best_match_idx] = true;

            // Update dominant color for matched track
            cv::Rect bbox_int = detections[best_match_idx].box;
            if (bbox_int.x >= 0 && bbox_int.y >= 0 && bbox_int.x + bbox_int.width <= frame.cols && bbox_int.y + bbox_int.height <= frame.rows) {
                cv::Mat player_roi = frame(bbox_int);
                tracks_[i].dominant_color = get_dominant_color(player_roi);
            }
        }
    }

    // 3. Remove stale tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [this](const Track& track) {
            return track.frames_since_update > max_frames_to_skip_;
        }), tracks_.end());

    // 4. Create new tracks for unmatched detections
    for (int i = 0; i < detections.size(); ++i) {
        if (!matched_detections[i]) {
            Track new_track;
            new_track.id = next_track_id_++;
            cv::Point2f detection_center(detections[i].box.x + detections[i].box.width / 2.0f, 
                                       detections[i].box.y + detections[i].box.height);
            new_track.kf.init(detection_center);
            new_track.last_bbox = detections[i].box;
            new_track.frames_since_update = 0;

            // Get dominant color for new track
            cv::Rect bbox_int = detections[i].box;
            if (bbox_int.x >= 0 && bbox_int.y >= 0 && bbox_int.x + bbox_int.width <= frame.cols && bbox_int.y + bbox_int.height <= frame.rows) {
                cv::Mat player_roi = frame(bbox_int);
                new_track.dominant_color = get_dominant_color(player_roi);
            } else {
                new_track.dominant_color = cv::Scalar(0,0,0); // Default to black if ROI is invalid
            }
            tracks_.push_back(new_track);
        }
    }
}

std::vector<std::pair<int, cv::Point2f>> PlayerTracker::get_tracks() {
    std::vector<std::pair<int, cv::Point2f>> current_tracks;
    for (const auto& track : tracks_) {
        // Return the smoothed position from the Kalman Filter
        cv::Point2f pos = track.kf.get_state();
        current_tracks.emplace_back(track.id, pos);
    }
    return current_tracks;
}

void PlayerTracker::assign_teams() {
    if (tracks_.empty()) {
        return;
    }

    // Collect all dominant colors
    cv::Mat all_colors_hsv(tracks_.size(), 3, CV_32F);
    std::map<int, int> track_id_to_row_idx; // Map track ID to its row in all_colors_hsv
    for (size_t i = 0; i < tracks_.size(); ++i) {
        all_colors_hsv.at<float>(i, 0) = tracks_[i].dominant_color[0]; // H
        all_colors_hsv.at<float>(i, 1) = tracks_[i].dominant_color[1]; // S
        all_colors_hsv.at<float>(i, 2) = tracks_[i].dominant_color[2]; // V
        track_id_to_row_idx[tracks_[i].id] = i;
    }

    // Determine number of clusters (K)
    // Assuming 2 teams + 1 referee = 3 clusters, or 2 teams if no referee
    int K = std::min((int)tracks_.size(), 3); 
    if (K < 2) { // Need at least 2 clusters for 2 teams
        K = tracks_.size(); // Assign each player to their own "team" if less than 2
        if (K == 0) return;
    }

    cv::Mat labels;
    cv::Mat centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);
    cv::kmeans(all_colors_hsv, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

    // Analyze clusters and assign team labels
    std::map<int, std::vector<int>> cluster_to_track_ids;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        int cluster_label = labels.at<int>(i);
        cluster_to_track_ids[cluster_label].push_back(tracks_[i].id);
    }

    // Sort clusters by size (number of players)
    std::vector<std::pair<int, std::vector<int>>> sorted_clusters;
    for (auto const& [cluster_label, track_ids] : cluster_to_track_ids) {
        sorted_clusters.push_back({(int)track_ids.size(), track_ids});
    }
    std::sort(sorted_clusters.rbegin(), sorted_clusters.rend()); // Sort in descending order of size

    // Assign team labels
    team_assignments_.clear();
    std::set<std::string> assigned_labels; // To ensure unique labels like "Team A", "Team B", "Referee"

    for (size_t i = 0; i < sorted_clusters.size(); ++i) {
        const auto& cluster_info = sorted_clusters[i];
        const std::vector<int>& track_ids = cluster_info.second;
        std::string current_label = "Unknown";

        if (i == 0 && assigned_labels.find("Team A") == assigned_labels.end()) {
            current_label = "Team A";
            assigned_labels.insert("Team A");
        } else if (i == 1 && assigned_labels.find("Team B") == assigned_labels.end()) {
            current_label = "Team B";
            assigned_labels.insert("Team B");
        } else if (track_ids.size() <= 2 && assigned_labels.find("Referee") == assigned_labels.end()) {
            // Heuristic for Referee: a very small cluster (1 or 2 players)
            current_label = "Referee";
            assigned_labels.insert("Referee");
        } else {
            // For any other clusters, assign "Unknown" or a generic "Team X" if more teams are expected
            // For simplicity, we'll stick to "Unknown" for remaining small clusters
            current_label = "Unknown";
        }

        for (int track_id : track_ids) {
            team_assignments_[track_id] = current_label;
        }
    }
}