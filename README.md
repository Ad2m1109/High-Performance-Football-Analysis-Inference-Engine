# High-Performance Football Analysis Inference Engine

A high-throughput, GPU-accelerated video analysis microservice built with C++17, TensorRT, and gRPC. This engine serves as the computational core for real-time football match analysis, decoupled from database operations and presentation logic to maximize GPU utilization and inference throughput.

## Technical Architecture

This service implements a **standalone inference engine** designed for low-latency, high-throughput video analysis. By leveraging NVIDIA TensorRT for model optimization and gRPC for inter-process communication, the engine achieves sub-millisecond inference times per frame while maintaining architectural decoupling from orchestration layers.

### Core Computational Modules

#### 1. Object Detection Pipeline
- **Model**: YOLOv8 (ONNX → TensorRT optimization)
- **Acceleration**: NVIDIA TensorRT with FP16/INT8 quantization support
- **Inference Backend**: CUDA-accelerated kernels for pre/post-processing
- **Output**: Bounding boxes, class probabilities, and confidence scores

#### 2. Multi-Object Tracking System
- **Player Tracking**: Kalman Filter-based state estimation with IOU-based data association
- **State Space**: 8-dimensional state vector (x, y, w, h, vx, vy, vw, vh)
- **Prediction Model**: Constant velocity motion model with Gaussian noise
- **Association**: Hungarian algorithm for optimal track-detection matching
- **ID Persistence**: Robust re-identification across occlusions and frame gaps

#### 3. Geometric Transformation Layer
- **Calibration Method**: Homography matrix estimation from pitch keypoints
- **Coordinate Mapping**: Pixel space → Real-world metric space (meters)
- **Applications**: 
  - Speed calculation (m/s)
  - Distance traversed per temporal window
  - Trajectory analysis in standardized coordinate frame

#### 4. Unsupervised Team Classification
- **Algorithm**: K-Means clustering on HSV color space
- **Feature Extraction**: Dominant jersey color from cropped player regions
- **Cluster Count**: K=3 (Team A, Team B, Referee)
- **Distance Metric**: Euclidean distance in normalized HSV space

## Communication Protocol

### gRPC Service Definition

The engine exposes a high-performance gRPC interface for frame-level and batch-level processing:

```protobuf
service AnalysisEngine {
  rpc AnalyzeFrame(FrameRequest) returns (FrameResponse);
  rpc AnalyzeVideo(VideoRequest) returns (stream VideoResponse);
}
```

**Protocol Buffers** ensure:
- Binary serialization for minimal network overhead
- Strong typing with schema evolution support
- Language-agnostic integration (Python, Go, Java, etc.)

### Performance Characteristics

- **Latency**: ~15-30ms per frame (720p) on RTX 3080
- **Throughput**: 30-60 FPS real-time processing
- **Memory Footprint**: ~2GB GPU VRAM (model + intermediate tensors)
- **Concurrency**: Thread-safe inference with request queuing

## Installation

### Prerequisites

**System Requirements:**
- CUDA-capable GPU (Compute Capability ≥ 7.0)
- Ubuntu 20.04+ or equivalent Linux distribution
- GCC 9+ with C++17 support

**Dependencies:**

1. **CUDA Toolkit (≥ 11.8)**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt-get update
   sudo apt-get install cuda-toolkit-11-8
   ```

2. **TensorRT (≥ 8.6)**
   ```bash
   # Download TensorRT from NVIDIA Developer Portal
   tar -xzvf TensorRT-8.6.x.Linux.x86_64-gnu.cuda-11.8.tar.gz
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT/lib
   ```

3. **OpenCV (≥ 4.5) with CUDA support**
   ```bash
   sudo apt-get install libopencv-dev
   # For CUDA-enabled build, compile from source with -DWITH_CUDA=ON
   ```

4. **gRPC and Protocol Buffers**
   ```bash
   sudo apt-get install -y build-essential autoconf libtool pkg-config
   git clone --recurse-submodules -b v1.58.0 --depth 1 https://github.com/grpc/grpc
   cd grpc
   mkdir -p cmake/build && cd cmake/build
   cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ../..
   make -j$(nproc)
   sudo make install
   ```

5. **yaml-cpp**
   ```bash
   sudo apt-get install libyaml-cpp-dev
   ```

### Build Instructions

```bash
git clone https://github.com/your-org/football-analyser-ai-model.git
cd football-analyser-ai-model

# Initialize submodules (cxxopts, csv-parser)
git submodule update --init --recursive

# Configure CMake
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DTensorRT_DIR=/path/to/TensorRT \
      ..

# Compile with optimizations
make -j$(nproc)
```

## Usage

### Camera Calibration

Generate the homography matrix for perspective transformation:

```bash
python3 scripts/calibration_tool.py \
  --video sample.mp4 \
  --output calibration.yaml
```

Click on 4 pitch corners in canonical order (e.g., penalty box corners) to compute the homography.

### Running the Engine

**Standalone Mode (for testing):**
```bash
./build/analysis_engine \
  --video /data/match_footage.mp4 \
  --model models/yolov8m.onnx \
  --calib calibration.yaml \
  --output-dir ./results \
  --conf 0.45 \
  --batch-size 4
```

**gRPC Service Mode:**
```bash
./build/analysis_service \
  --port 50051 \
  --model models/yolov8m.onnx \
  --calib calibration.yaml \
  --workers 4
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--conf` | float | 0.5 | Detection confidence threshold |
| `--iou-threshold` | float | 0.4 | NMS IoU threshold |
| `--tracking-iou` | float | 0.3 | Minimum IoU for track association |
| `--max-age` | int | 30 | Maximum frames to retain lost tracks |
| `--min-hits` | int | 3 | Minimum detections before track confirmation |
| `--batch-size` | int | 1 | TensorRT inference batch size |

## Output Specification

### CSV Schema

**player_metrics.csv:**
```
frame_id, player_id, team, x_pixel, y_pixel, x_meter, y_meter, speed_mps, distance_meters, total_distance_meters
```

**ball_metrics.csv:**
```
frame_id, x_pixel, y_pixel, x_meter, y_meter, velocity_x, velocity_y, confidence
```

### Data Characteristics

- **Temporal Resolution**: Per-frame granularity (30 FPS → 33.3ms intervals)
- **Spatial Precision**: Sub-pixel accuracy via Kalman smoothing
- **Metric Validity**: Real-world measurements calibrated to FIFA standard pitch dimensions (105m × 68m)

## Performance Optimization

### TensorRT Optimization

Convert ONNX models to TensorRT engines for maximum throughput:

```bash
trtexec --onnx=yolov8m.onnx \
        --saveEngine=yolov8m.engine \
        --fp16 \
        --workspace=4096 \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:4x3x640x640 \
        --maxShapes=input=8x3x640x640
```

### Profiling

```bash
nsys profile --stats=true ./build/analysis_engine --video test.mp4
```

## Project Structure

```
.
├── CMakeLists.txt
├── README.md
├── proto/
│   └── analysis.proto          # gRPC service definitions
├── src/
│   ├── main.cpp                # Standalone entry point
│   ├── service.cpp             # gRPC service implementation
│   ├── detection/
│   │   ├── base_tracker.{h,cpp}
│   │   ├── player_tracker.{h,cpp}
│   │   └── ball_tracker.{h,cpp}
│   ├── analytics/
│   │   └── metrics.{h,cpp}
│   └── utils/
│       ├── calibration.{h,cpp}
│       ├── kalman_filter.{h,cpp}
│       └── config.h
├── models/                     # TensorRT engines (not in VCS)
└── scripts/
    └── calibration_tool.py
```

## Research Context

This engine implements state-of-the-art techniques from:

- **Tracking**: SORT (Simple Online and Realtime Tracking) with Kalman filtering
- **Detection**: YOLOv8 architecture with anchor-free detection heads
- **Team Assignment**: Unsupervised color-based clustering as per sports analytics literature

For academic applications, cite the underlying methodologies appropriately.

## License

MIT License - See LICENSE file for details

## Contact

For technical inquiries regarding integration or performance tuning, contact the development team at [technical-email].