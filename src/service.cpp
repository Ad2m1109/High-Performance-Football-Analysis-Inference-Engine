#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "analysis.grpc.pb.h"
#include <grpcpp/grpcpp.h>

using analysis::AnalysisEngine;
using analysis::AnalysisResult;
using analysis::VideoRequest;
using analysis::VideoResponse;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

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
    response.set_message("Job queued");
    writer->Write(response);

    // 2. Simulate processing: PROCESSING
    for (int i = 1; i <= 10; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1)); // Simulate work

      response.set_status("PROCESSING");
      response.set_progress(i * 0.1);
      response.set_message("Analyzing frames...");
      writer->Write(response);

      if (context->IsCancelled()) {
        return Status::CANCELLED;
      }
    }

    // 3. Final response: COMPLETED
    response.set_status("COMPLETED");
    response.set_progress(1.0);
    response.set_message("Analysis finished successfully");

    AnalysisResult *result = response.mutable_result();
    result->set_match_id(request->match_id());
    result->set_total_frames(300); // Example
    result->set_players_tracked(22);
    result->set_report_id("report_" + request->match_id());

    // In a real implementation, these would be the actual paths to generated
    // CSVs
    result->set_player_metrics_csv_path("/tmp/player_metrics.csv");
    result->set_ball_metrics_csv_path("/tmp/ball_metrics.csv");

    writer->Write(response);

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
  std::cout << "Server listening on " << server_address << std::endl;
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
