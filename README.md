# Football Analysis Tool

This project is a C++-based tool for analyzing football match videos. It uses computer vision and deep learning with TensorRT acceleration to detect and track players and the ball, assign players to teams, and calculate various performance metrics.

## Features

*   **High-Performance Detection:** Utilizes TensorRT-accelerated models for efficient player and ball detection in video frames.
*   **Robust Tracking:** Employs **Kalman Filters and IOU matching** to track players and the ball, maintaining consistent IDs even through occlusions and providing smoother trajectories.
*   **Automatic Team Assignment:** Assigns players to teams (Team A, Team B) by analyzing the **dominant color of their jerseys using K-Means clustering**, also identifying referees.
*   **Real-World Metrics:** Leverages a homography-based camera calibration to convert pixel coordinates to real-world coordinates (meters), enabling accurate calculation of:
    *   Player speed (in meters per second - `speed_mps`).
    *   Distance covered per frame (in meters - `distance_meters`).
    *   Total cumulative distance covered by each player (in meters - `total_distance_meters`).
*   **CSV Export:** Exports all calculated data to CSV files for further analysis and visualization.

## Project Structure

```
/
├── CMakeLists.txt
├── README.md
├── build/
├── cxxopts/
├── fast-cpp-csv-parser/
└── src/
    ├── main.cpp
    ├── analytics/
    │   ├── metrics.cpp
    │   └── metrics.h
    ├── detection/
    │   ├── ball_tracker.cpp
    │   ├── ball_tracker.h
    │   ├── base_tracker.cpp
    │   ├── base_tracker.h
    │   ├── player_tracker.cpp
    │   └── player_tracker.h
    └── utils/
        ├── calibration.cpp
        ├── calibration.h
        ├── config.h
        ├── kalman_filter.cpp
        └── kalman_filter.h
```

### Component Breakdown

*   `src/main.cpp`: The main entry point of the application. It handles command-line argument parsing, initializes the trackers and metric calculator, and orchestrates the video processing pipeline.
*   `src/detection/`: This module contains the core logic for object detection and tracking.
    *   `base_tracker.h/.cpp`: An abstract base class that handles the TensorRT inference engine.
    *   `player_tracker.h/.cpp`: Implements player detection, **robust tracking using Kalman filters and IOU matching**, and **team assignment based on dominant color analysis**.
    *   `ball_tracker.h/.cpp`: Implements ball detection and **smoother tracking using a Kalman filter**, including handling occlusions by players.
*   `src/analytics/`: This module is responsible for calculating high-level performance metrics.
    *   `metrics.h/.cpp`: Calculates player speeds, distances per frame, and total cumulative distances using the real-world coordinate data and team assignments.
*   `src/utils/`: Contains essential utilities for the project.
    *   `calibration.h/.cpp`: Manages the camera calibration, loading the homography matrix from a YAML file and converting between pixel and real-world coordinates.
    *   `kalman_filter.h/.cpp`: Implements a **Kalman filter** for object state estimation and smoothing.
    *   `config.h`: A struct to hold all configuration parameters.
*   `cxxopts/` & `fast-cpp-csv-parser/`: Included third-party libraries for command-line option parsing and efficient CSV file writing.
*   `CMakeLists.txt`: The build script for the project. It defines dependencies (OpenCV, TensorRT, CUDA, YAML-CPP) and compilation settings.

## Setup and Installation

1.  **Prerequisites:**
    *   CMake (>= 3.16)
    *   A C++17 compatible compiler (e.g., GCC, Clang)
    *   NVIDIA CUDA Toolkit
    *   NVIDIA TensorRT
    *   OpenCV
    *   `yaml-cpp` library

2.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3.  **Build the project:**
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

## How to Run

1.  **Camera Calibration (Important!):**

    Before running the analysis, you **must** have a `calibration.yaml` file. This file contains the homography matrix required to map pixel coordinates to real-world pitch coordinates. The application will look for this file based on the path provided in the command-line arguments.

    You can use a Python script to help generate this file (assuming you have Python and OpenCV installed):
    ```bash
    python3 src/utils/calibration.py
    ```
    This script will guide you through clicking points on a video frame to calculate the homography.

2.  **Run the main analysis pipeline:**

    The compiled executable will be located in the `build` directory.
    ```bash
    ./build/test_runner --video /path/to/your/video.mp4 --calib /path/to/your/calibration.yaml --model /path/to/your/yolov8m.onnx --output-dir /path/to/save/results
    ```
    
    **Command-Line Arguments:**
    *   `-v, --video`: (Required) Path to the input video file.
    *   `-c, --calib`: (Required) Path to the camera calibration YAML file.
    *   `-m, --model`: (Required) Path to the YOLOv8 ONNX model file (e.g., `yolov8m.onnx`).
    *   `-o, --output-dir`: Directory to save the output CSV files (defaults to the current directory).
    *   `--conf`: Confidence threshold for object detection (default: 0.5).
    *   `--no-ball`: Disable ball tracking (default: enabled).
    *   `-h, --help`: Print usage information.

## Output

The primary output of the tool is a set of CSV files saved in the specified output directory:

*   `player_metrics.csv`: Contains detailed information for each player in every frame, including their consistent `player_id`, real-world coordinates (`x`, `y`), **assigned `team`**, **`speed_mps`**, **`distance_meters`**, and **`total_distance_meters`**.
*   `ball_metrics.csv`: If ball tracking is enabled, this file contains the detected and smoothed position of the ball in each frame, including its real-world coordinates.

This data can be used for a wide range of downstream tasks, such as creating frontend visualizations (heatmaps, trajectories), in-depth statistical analysis, or tactical analysis of team formations and player movements.