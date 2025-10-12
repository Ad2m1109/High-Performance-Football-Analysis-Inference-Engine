#include "analytics/metrics.h"
#include "csv.h"
#include <fstream>
#include <iostream>
#include <cmath>

MetricsCalculator::MetricsCalculator(const std::string& output_dir) : output_dir_(output_dir) {}

void MetricsCalculator::process_frame(int frame_count, double fps, const std::vector<std::pair<int, cv::Point2f>>& player_tracks, const std::pair<int, cv::Point2f>& ball_track, const std::map<int, std::string>& team_assignments) {
    // Update fps (in case frames are processed with different sources)
    video_fps_ = fps > 0.0 ? fps : video_fps_;

    // Process player metrics
    for (const auto& track : player_tracks) {
        std::map<std::string, std::string> player_metric;
        player_metric["frame"] = std::to_string(frame_count);
        player_metric["player_id"] = std::to_string(track.first);
        player_metric["x"] = std::to_string(track.second.x);
        player_metric["y"] = std::to_string(track.second.y);

        // Add team assignment
        auto it_team = team_assignments.find(track.first);
        if (it_team != team_assignments.end()) {
            player_metric["team"] = it_team->second;
        } else {
            player_metric["team"] = "Unknown";
        }

        // Calculate speed and distance
        double speed_mps = 0.0;
        double distance_meters = 0.0;
        double total_distance_meters = 0.0;

        auto it_last_pos = last_player_positions_.find(track.first);
        auto it_last_frame = last_player_frame_counts_.find(track.first);

        if (it_last_pos != last_player_positions_.end() && it_last_frame != last_player_frame_counts_.end()) {
            // Calculate distance covered in this frame interval
            distance_meters = cv::norm(track.second - it_last_pos->second);
            
            // Calculate time elapsed between last processed frame and current frame
            double delta_frames = frame_count - it_last_frame->second;
            double delta_time = delta_frames / fps;

            // Calculate speed (meters per second)
            if (delta_time > 0) {
                speed_mps = distance_meters / delta_time;
            }

            // Update total distance
            player_total_distances_[track.first] += distance_meters;
            total_distance_meters = player_total_distances_[track.first];
        } else {
            // Initialize total distance for new player
            player_total_distances_[track.first] = 0.0;
        }

        player_metric["speed_mps"] = std::to_string(speed_mps);
        player_metric["distance_meters"] = std::to_string(distance_meters);
        player_metric["total_distance_meters"] = std::to_string(total_distance_meters);

        // Increment frame count seen for this player
        player_frame_counts_[track.first] += 1;

        // Fill DB-oriented placeholder fields (events detection not implemented here)
        // These defaults match the schema and will be persisted as zeros/nulls
        player_metric["minutes_played"] = ""; // computed at export time
        player_metric["shots"] = "0";
        player_metric["shots_on_target"] = "0";
        player_metric["passes"] = "0";
        player_metric["accurate_passes"] = "0";
        player_metric["tackles"] = "0";
        player_metric["interceptions"] = "0";
        player_metric["clearances"] = "0";
        player_metric["saves"] = "0";
        player_metric["fouls_committed"] = "0";
        player_metric["fouls_suffered"] = "0";
        player_metric["offsides"] = "0";
        player_metric["player_xg"] = "0.0";
        player_metric["key_passes"] = "0";
        player_metric["progressive_carries"] = "0";
        player_metric["press_resistance_success_rate"] = "0.0";
        player_metric["defensive_coverage_km"] = "0.0";
        player_metric["notes"] = "";
        player_metric["rating"] = "0.0";

        // Update last position and frame count for next frame's calculation
        last_player_positions_[track.first] = track.second;
        last_player_frame_counts_[track.first] = frame_count;

        player_metrics_.push_back(player_metric);
    }

    // Process ball metrics
    if (ball_track.first != -1) {
        std::map<std::string, std::string> ball_metric;
        ball_metric["frame"] = std::to_string(frame_count);
        ball_metric["x"] = std::to_string(ball_track.second.x);
        ball_metric["y"] = std::to_string(ball_track.second.y);
        ball_metrics_.push_back(ball_metric);
    }
}

void MetricsCalculator::save_to_csv() {
    // Save player metrics
    if (!player_metrics_.empty()) {
        std::ofstream file(output_dir_ + "/player_metrics.csv");
        if (file.is_open()) {
            // Write header in a stable DB-friendly order
            std::vector<std::string> header = {
                "frame",
                "player_id",
                "x",
                "y",
                "team",
                "minutes_played",
                "shots",
                "shots_on_target",
                "passes",
                "accurate_passes",
                "tackles",
                "interceptions",
                "clearances",
                "saves",
                "fouls_committed",
                "fouls_suffered",
                "offsides",
                "distance_meters",
                "total_distance_meters",
                "distance_covered_km",
                "player_xg",
                "key_passes",
                "progressive_carries",
                "press_resistance_success_rate",
                "defensive_coverage_km",
                "notes",
                "rating"
            };

            // write header
            for (size_t i = 0; i < header.size(); ++i) {
                if (i) file << ",";
                file << header[i];
            }
            file << std::endl;

            // Aggregate per-player totals to compute minutes and km and then write rows
            // We'll compute minutes_played per player based on frames seen and video_fps_
            std::map<int, int> frames_seen;
            for (const auto& m : player_metrics_) {
                int pid = std::stoi(m.at("player_id"));
                frames_seen[pid] = player_frame_counts_[pid];
            }

            for (const auto& m : player_metrics_) {
                int pid = std::stoi(m.at("player_id"));
                // To avoid duplicate player-level rows, we will still emit per-frame rows
                // but fill minutes and distance_covered_km using cumulative data for that player.
                // distance_covered_km = total_distance_meters / 1000.0
                double total_m = player_total_distances_[pid];
                double dist_km = total_m / 1000.0;
                int minutes = 0;
                if (video_fps_ > 0 && frames_seen[pid] > 0) {
                    double seconds = static_cast<double>(frames_seen[pid]) / video_fps_;
                    minutes = static_cast<int>(seconds / 60.0);
                }

                // Write values in header order
                std::vector<std::string> row_vals;
                row_vals.push_back(m.at("frame"));
                row_vals.push_back(m.at("player_id"));
                row_vals.push_back(m.at("x"));
                row_vals.push_back(m.at("y"));
                row_vals.push_back(m.at("team"));
                row_vals.push_back(std::to_string(minutes));
                row_vals.push_back(m.at("shots"));
                row_vals.push_back(m.at("shots_on_target"));
                row_vals.push_back(m.at("passes"));
                row_vals.push_back(m.at("accurate_passes"));
                row_vals.push_back(m.at("tackles"));
                row_vals.push_back(m.at("interceptions"));
                row_vals.push_back(m.at("clearances"));
                row_vals.push_back(m.at("saves"));
                row_vals.push_back(m.at("fouls_committed"));
                row_vals.push_back(m.at("fouls_suffered"));
                row_vals.push_back(m.at("offsides"));
                // distance_meters
                row_vals.push_back(m.at("distance_meters"));
                // total_distance_meters
                row_vals.push_back(m.at("total_distance_meters"));
                // distance_covered_km
                row_vals.push_back(std::to_string(dist_km));
                row_vals.push_back(m.at("player_xg"));
                row_vals.push_back(m.at("key_passes"));
                row_vals.push_back(m.at("progressive_carries"));
                row_vals.push_back(m.at("press_resistance_success_rate"));
                row_vals.push_back(m.at("defensive_coverage_km"));
                row_vals.push_back(m.at("notes"));
                row_vals.push_back(m.at("rating"));

                for (size_t i = 0; i < row_vals.size(); ++i) {
                    if (i) file << ",";
                    file << row_vals[i];
                }
                file << std::endl;
            }
            file.close();
        } else {
            std::cerr << "Error: Could not open player_metrics.csv for writing." << std::endl;
        }
    }

    // Save ball metrics
    if (!ball_metrics_.empty()) {
        std::ofstream file(output_dir_ + "/ball_metrics.csv");
        if (file.is_open()) {
            // Write header
            bool first_col = true;
            for (auto const& [key, val] : ball_metrics_[0]) {
                if (!first_col) file << ",";
                file << key;
                first_col = false;
            }
            file << std::endl;

            // Write data
            for (const auto& metric : ball_metrics_) {
                bool first_col = true;
                for (auto const& [key, val] : metric) {
                    if (!first_col) file << ",";
                    file << val;
                    first_col = false;
                }
                file << std::endl;
            }
            file.close();
        } else {
            std::cerr << "Error: Could not open ball_metrics.csv for writing." << std::endl;
        }
    }
}
