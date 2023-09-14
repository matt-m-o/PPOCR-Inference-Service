cmake_minimum_required(VERSION 3.5.1)
set (VCPKG_HOME $ENV{VCPKG_HOME}) #IMPORTANT: change this to where your vcpkg exists.
set (CMAKE_TOOLCHAIN_FILE "${VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake")
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE Debug)

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
include(${CMAKE_SOURCE_DIR}/src/service_http/CMakeLists.txt)
include(${CMAKE_SOURCE_DIR}/src/service_grpc/CMakeLists.txt)