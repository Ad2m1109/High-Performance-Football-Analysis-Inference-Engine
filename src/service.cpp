#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

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

  Status StreamAnalysis(
      ServerContext *context,
      grpc::ServerReaderWriter<analysis::MetricsUpdate, analysis::VideoChunk>
          *stream) override {
    std::cout << "Starting real-time streaming analysis..." << std::endl;

    analysis::VideoChunk first_chunk;
    if (!stream->Read(&first_chunk)) {
      return Status(grpc::StatusCode::INVALID_ARGUMENT, "No data received");
    }

    std::string match_id = first_chunk.match_id();
    std::string fifo_path = "/tmp/analysis_fifo_" + match_id;
    mkfifo(fifo_path.c_str(), 0666);

    std::atomic<bool> streaming_done{false};
    std::atomic<bool> error_occurred{false};
    std::string error_msg = "";

    // Thread to feed chunks to FIFO
    std::thread feeder_thread([&]() {
      int fd = open(fifo_path.c_str(), O_WRONLY);
      if (fd == -1) {
        error_occurred = true;
        error_msg = "Failed to open FIFO for writing";
        return;
      }

      // Process first chunk
      write(fd, first_chunk.data().data(), first_chunk.data().size());

      analysis::VideoChunk chunk;
      while (stream->Read(&chunk)) {
        write(fd, chunk.data().data(), chunk.data().size());
        if (chunk.is_last_chunk())
          break;
      }
      close(fd);
      streaming_done = true;
    });

    try {
      // Initialize Components (Same as AnalyzeVideo but streaming)
      Calibration calibration(first_chunk.calibration_path());
      YoloV8 yolo_detector(first_chunk.model_path().empty()
                               ? "yolov8m.onnx"
                               : first_chunk.model_path());
      PlayerTracker player_tracker;
      BallTracker ball_tracker;

      // Output dir for final artifacts if needed
      std::string output_dir = "/tmp/analysis_stream_" + match_id;
      fs::create_directories(output_dir);

      cv::VideoCapture cap(fifo_path);
      if (!cap.isOpened()) {
        streaming_done = true; // Stop thread
        feeder_thread.join();
        unlink(fifo_path.c_str());
        return Status(grpc::StatusCode::INTERNAL,
                      "Failed to open video stream via FIFO");
      }

      cv::Mat frame;
      int current_frame_idx = 0;
      double video_fps = cap.get(cv::CAP_PROP_FPS);
      if (video_fps == 0)
        video_fps = 30.0;

      while (cap.read(frame)) {
        current_frame_idx++;

        // 1. Detection
        auto all_detections = yolo_detector.detect(frame);
        std::vector<Detection> player_detections, ball_detections;
        for (const auto &det : all_detections) {
          if (det.class_id == 0)
            player_detections.push_back(det);
          else if (det.class_id == 32)
            ball_detections.push_back(det);
        }

        // 2. Tracking
        player_tracker.update(player_detections, frame);
        ball_tracker.update(ball_detections);

        // 3. Real-world Projection
        auto real_world_players =
            calibration.transform(player_tracker.get_tracks());
        auto real_world_ball = calibration.transform(ball_tracker.get_track());

        // 4. Send metrics back in real-time
        if (current_frame_idx % 5 == 0) { // Throttling updates to 6Hz approx
          analysis::MetricsUpdate update;
          update.set_status("PROCESSING");
          update.set_message("Processing frame " +
                             std::to_string(current_frame_idx));

          // Add Player Metrics
          for (const auto &player_pair : real_world_players) {
            auto *m = update.add_metrics();
            m->set_player_id(player_pair.first);
            m->set_x(player_pair.second.x);
            m->set_y(player_pair.second.y);
            m->set_frame_index(current_frame_idx);
          }

          // Add Ball Metric
          if (real_world_ball.first != -1) {
            auto *b = update.mutable_ball_metric();
            b->set_x(real_world_ball.second.x);
            b->set_y(real_world_ball.second.y);
            b->set_frame_index(current_frame_idx);
          }

          stream->Write(update);
        }
      }

      feeder_thread.join();
      unlink(fifo_path.c_str());

      analysis::MetricsUpdate final_update;
      final_update.set_status("COMPLETED");
      final_update.set_progress(1.0);
      final_update.set_message("Real-time analysis finished. " +
                               std::to_string(current_frame_idx) +
                               " frames processed.");
      stream->Write(final_update);

    } catch (const std::exception &e) {
      if (feeder_thread.joinable())
        feeder_thread.join();
      unlink(fifo_path.c_str());
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
