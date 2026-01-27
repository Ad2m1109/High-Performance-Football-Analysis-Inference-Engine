cmake_minimum_required(VERSION 3.16)
project(SportsAnalytics)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# Find Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

# Find gRPC using PkgConfig
pkg_check_modules(GRPC REQUIRED grpc++)
message(STATUS "Using gRPC via PkgConfig")

# Find gRPC C++ plugin
find_program(GRPC_CPP_PLUGIN_EXECUTABLE NAMES grpc_cpp_plugin)
if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "grpc_cpp_plugin not found!")
endif()

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

# Protobuf generation
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

set(PROTO_FILE ${PROTO_SRC_DIR}/analysis.proto)
set(PROTO_PB_H "${PROTO_OUT_DIR}/analysis.pb.h")
set(PROTO_PB_CC "${PROTO_OUT_DIR}/analysis.pb.cc")
set(PROTO_GRPC_PB_H "${PROTO_OUT_DIR}/analysis.grpc.pb.h")
set(PROTO_GRPC_PB_CC "${PROTO_OUT_DIR}/analysis.grpc.pb.cc")

add_custom_command(
    OUTPUT "${PROTO_PB_H}" "${PROTO_PB_CC}" "${PROTO_GRPC_PB_H}" "${PROTO_GRPC_PB_CC}"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
         --cpp_out "${PROTO_OUT_DIR}"
         --grpc_out "${PROTO_OUT_DIR}"
         -I "${PROTO_SRC_DIR}"
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${YAMLCPP_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fast-cpp-csv-parser)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROTO_OUT_DIR})
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${GRPC_INCLUDE_DIRS})

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

# Source files for gRPC service
set(SERVICE_SOURCES
    src/service.cpp
    src/analytics/metrics.cpp
    src/detection/player_tracker.cpp
    src/detection/ball_tracker.cpp
    src/detection/yolov8.cpp
    src/utils/calibration.cpp
    src/utils/logger.cpp
    src/utils/kalman_filter.cpp
    "${PROTO_PB_CC}"
    "${PROTO_GRPC_PB_CC}"
)

# Create executables
add_executable(test_runner ${SOURCES})
add_executable(analysis_service ${SERVICE_SOURCES})

# Set RPATH for build directory
set_target_properties(test_runner analysis_service PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN:/usr/lib/x86_64-linux-gnu")

# Link libraries
set(COMMON_LIBS
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
    nvinfer
    nvonnxparser
)

target_link_libraries(test_runner ${COMMON_LIBS})
target_link_libraries(analysis_service 
    ${COMMON_LIBS}
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
)

# Compiler definitions
target_compile_definitions(test_runner PRIVATE ${YAMLCPP_CFLAGS_OTHER})
target_compile_definitions(analysis_service PRIVATE ${YAMLCPP_CFLAGS_OTHER})

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(OPT_FLAGS -O3 -DNDEBUG)
else()
    set(OPT_FLAGS -g -O0)
endif()

target_compile_options(test_runner PRIVATE ${OPT_FLAGS})
target_compile_options(analysis_service PRIVATE ${OPT_FLAGS})

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "YAML-CPP found: ${YAMLCPP_FOUND}")
