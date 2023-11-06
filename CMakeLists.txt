cmake_minimum_required(VERSION 3.5.1)
set (VCPKG_HOME $ENV{VCPKG_ROOT}) #IMPORTANT: change this to where your vcpkg exists.
set (CMAKE_TOOLCHAIN_FILE "${VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake")
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE Release) # Debug / Release
set (CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O2")

project(PPOCR-Inference-Service)


option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})


find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(Threads REQUIRED)

# nlohmann json
set(NLOHMANN_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/nlohmann)

# Base64
set(BASE64_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/cpp-base64-2.rc.08)

# Includes (FirstParty Libs)
set(INCLUDES_DIR ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(protos)
# include(${CMAKE_SOURCE_DIR}/src/service_http/CMakeLists.txt)
include(${CMAKE_SOURCE_DIR}/src/service_grpc/CMakeLists.txt)

if (UNIX)
    set(CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}/Release")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")

    install( DIRECTORY
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/paddle_inference/third_party/install/mkldnn/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/paddle_inference/third_party/install/mklml/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/paddle_inference/paddle/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/paddle2onnx/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/openvino/runtime/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/openvino/runtime/3rdparty/omp/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/onnxruntime/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/fast_tokenizer/lib/
        /workspaces/PPOCR-Inference-Service/sdk/fastdeploy-linux-x64-1.0.7/third_libs/install/opencv/lib64/

        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
endif()