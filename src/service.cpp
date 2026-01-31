#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "analysis.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include "analytics/metrics.h"
#include "detection/ball_tracker.h"
#include "detection/player_tracker.h"
#include "detection/yolov8.h"
#include "utils/calibration.h"
#include <opencv2/videoio.hpp>

using analysis::AnalysisEngine;
using analysis::AnalysisResult;
using analysis::VideoRequest;
using analysis::VideoResponse;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

namespace fs = std::filesystem;

class AnalysisEngineServiceImpl final : public AnalysisEngine::Service {
  Status AnalyzeVideo(ServerContext *context, const VideoRequest *request,
                      ServerWriter<VideoResponse> *writer) override {

    std::cout << "Received analysis request for match: " << request->match_id()
              << std::endl;
    std::cout << "Video path: " << request->video_path() << std::endl;

    // 1. Initial response: PENDING
    VideoResponse response;
    response.set_job_id(request->match_id());
    response.set_status("PENDING");
    response.set_progress(0.0);
    response.set_message("Initializing analysis engine...");
    writer->Write(response);

    try {
      // Initialize components
      Calibration calibration(request->calibration_path());

      // Use model path from request
      std::string model_path = request->model_path();
      if (model_path.empty()) {
        model_path = "yolov8m.onnx"; // Fallback
      }
      YoloV8 yolo_detector(model_path);

      PlayerTracker player_tracker;
      BallTracker ball_tracker;

      // Create a temporary output directory
      std::string output_dir = "/tmp/analysis_" + request->match_id();
      fs::create_directories(output_dir);
      MetricsCalculator metrics_calculator(output_dir);

      // Open video
      cv::VideoCapture cap(request->video_path());
      if (!cap.isOpened()) {
        response.set_status("FAILED");
        response.set_message("Could not open video file: " +
                             request->video_path());
        writer->Write(response);
        return Status(grpc::StatusCode::NOT_FOUND, "Video file not found");
      }

      double video_fps = cap.get(cv::CAP_PROP_FPS);
      if (video_fps == 0)
        video_fps = 30.0;

      int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
      cv::Mat frame;
      int current_frame_idx = 0;

      // 2. Processing loop
      while (cap.read(frame)) {
        current_frame_idx++;

        // Perform detection
        auto all_detections = yolo_detector.detect(frame);

        std::vector<Detection> player_detections;
        std::vector<Detection> ball_detections;
        for (const auto &det : all_detections) {
          if (det.class_id == 0 &&
              det.confidence >= request->confidence_threshold()) {
            player_detections.push_back(det);
          } else if (det.class_id == 32 &&
                     det.confidence >= request->confidence_threshold()) {
            ball_detections.push_back(det);
          }
        }

        // Update trackers
        player_tracker.update(player_detections, frame);
        ball_tracker.update(ball_detections);

        // Convert to real-world coordinates
        auto real_world_players =
            calibration.transform(player_tracker.get_tracks());
        auto real_world_ball = calibration.transform(ball_tracker.get_track());

        // Calculate metrics
        metrics_calculator.process_frame(current_frame_idx, video_fps,
                                         real_world_players, real_world_ball,
                                         player_tracker.get_team_assignments());

        // Send progress update every 30 frames
        if (current_frame_idx % 30 == 0) {
          float progress = static_cast<float>(current_frame_idx) / total_frames;
          response.set_status("PROCESSING");
          response.set_progress(progress);
          response.set_message("Processing frame " +
                               std::to_string(current_frame_idx) + "/" +
                               std::to_string(total_frames));
          writer->Write(response);
        }

        if (context->IsCancelled()) {
          return Status::CANCELLED;
        }
      }

      // 3. Finalize
      player_tracker.assign_teams();
      metrics_calculator.save_to_csv();

      // 4. Final response: COMPLETED
      response.set_status("COMPLETED");
      response.set_progress(1.0);
      response.set_message("Analysis finished successfully");

      AnalysisResult *result = response.mutable_result();
      result->set_match_id(request->match_id());
      result->set_total_frames(total_frames);
      result->set_players_tracked(player_tracker.get_tracks().size());
      result->set_report_id("report_" + request->match_id());
      result->set_player_metrics_csv_path(output_dir + "/player_metrics.csv");
      result->set_ball_metrics_csv_path(output_dir + "/ball_metrics.csv");

      writer->Write(response);

    } catch (const std::exception &e) {
      std::cerr << "Error during analysis: " << e.what() << std::endl;
      response.set_status("FAILED");
      response.set_message(std::string("Internal error: ") + e.what());
      writer->Write(response);
      return Status(grpc::StatusCode::INTERNAL, e.what());
    }

    return Status::OK;
  }
};

void RunServer(const std::string &port) {
  std::string server_address("0.0.0.0:" + port);
  AnalysisEngineServiceImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Analysis Service listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char **argv) {
  std::string port = "50051";
  if (argc > 1) {
    port = argv[1];
  }
  RunServer(port);
  return 0;
}
