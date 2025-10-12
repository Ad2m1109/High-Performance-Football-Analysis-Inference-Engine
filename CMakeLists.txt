cmake_minimum_required(VERSION 3.16)
project(SportsAnalytics)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# Find CUDA
find_package(CUDA REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/cuda/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/cuda/lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found!")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${YAMLCPP_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fast-cpp-csv-parser)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/analytics/metrics.cpp
    src/detection/player_tracker.cpp
    src/detection/ball_tracker.cpp
    src/detection/yolov8.cpp
    src/utils/calibration.cpp
    src/utils/logger.cpp
    src/utils/kalman_filter.cpp
)

# Set RPATH so the executable knows where to find TensorRT libraries at runtime
set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Create executable
add_executable(test_runner ${SOURCES})

# Also set the RPATH for the build directory
set_target_properties(test_runner PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON,
    INSTALL_RPATH "$ORIGIN:/usr/lib/x86_64-linux-gnu")

# Link libraries
target_link_libraries(test_runner
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
    nvinfer
    nvonnxparser
)

# Compiler definitions
target_compile_definitions(test_runner PRIVATE ${YAMLCPP_CFLAGS_OTHER})

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test_runner PRIVATE -O3 -DNDEBUG)
else()
    target_compile_options(test_runner PRIVATE -g -O0)
endif()

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "YAML-CPP found: ${YAMLCPP_FOUND}")
