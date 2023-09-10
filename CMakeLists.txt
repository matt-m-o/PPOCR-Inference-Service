cmake_minimum_required(VERSION 3.0)
project(PPOCR-Inference-Service)


option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

# Http
set(HTTP_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/cpp-httplib-0.14.0)

# nlohmann json
set(NLOHMANN_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/nlohmann)

# Base64
set(BASE64_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/cpp-base64-2.rc.08)

# Includes (FirstParty Libs)
set(INCLUDES_DIR ${CMAKE_SOURCE_DIR}/src)

# Executable
add_executable(ppocr_infer_service ${CMAKE_SOURCE_DIR}/src/main.cc)
target_include_directories(ppocr_infer_service PRIVATE ${HTTP_LIB_INCLUDE_DIR})
target_link_libraries(ppocr_infer_service ${FASTDEPLOY_LIBS})
target_link_libraries(ppocr_infer_service ${NLOHMANN_LIB_INCLUDE_DIR})
target_link_libraries(ppocr_infer_service ${BASE64_LIB_INCLUDE_DIR})
