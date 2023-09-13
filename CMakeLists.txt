cmake_minimum_required(VERSION 3.0)
project(PPOCR-Inference-Service)


option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})


# nlohmann json
set(NLOHMANN_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/nlohmann)

# Base64
set(BASE64_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/cpp-base64-2.rc.08)

# Includes (FirstParty Libs)
set(INCLUDES_DIR ${CMAKE_SOURCE_DIR}/src)


# add_subdirectory(src)
include(${CMAKE_SOURCE_DIR}/src/service_http/CMakeLists.txt)