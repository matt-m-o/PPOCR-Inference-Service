cmake_minimum_required(VERSION 3.0)
project(PPOCR-Inference-Service)


# Http
set(HTTP_LIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/includes/cpp-httplib-0.14.0)

# Executable
add_executable(ppocr_infer_service_http ${CMAKE_SOURCE_DIR}/src/service_http/service_http.cc)
target_include_directories(ppocr_infer_service_http PRIVATE ${HTTP_LIB_INCLUDE_DIR})
target_link_libraries(ppocr_infer_service_http ${FASTDEPLOY_LIBS})
target_link_libraries(ppocr_infer_service_http ${NLOHMANN_LIB_INCLUDE_DIR})
target_link_libraries(ppocr_infer_service_http ${BASE64_LIB_INCLUDE_DIR})
