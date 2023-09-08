#include <string>
#include "ppocr_infer.hpp"

// "DTO"
struct AppOptions {
  std::string detection_model_dir = "./ch_PP-OCRv3_det_infer";
  std::string classification_model_dir = "./ch_ppocr_mobile_v2.0_cls_infer";
  std::string recognition_model_dir = "./japan_PP-OCRv4_rec_infer";
  std::string recognition_label_file_dir = "./dict_japan.txt"; // Dictionary
  std::string inference_backend = "ONNX_CPU";
  int server_port = 12345;
};



struct Settings {
  fastdeploy::RuntimeOption runtime_option;
  Models models;
  int server_port;  
};



AppOptions handleAppArgs( int argc, char *argv[] ) {

  std::array<std::string, 8> const BACKENDS = { "Paddle_CPU", "Open_VINO", "ONNX_CPU", "Paddle_Lite", "Paddle_GPU", "Paddle_GPU_Tensor_RT", "ONNX_GPU", "Tensor_RT" };

  AppOptions app_options;

  if (argc < 6) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/cls_model "
                "path/to/rec_model path/to/rec_label_file path/to/image "
                "run_option, "
                "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
                "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer "
                "./ppocr_keys_v1.txt ./12.jpg 0 \n"
            << std::endl;
    std::cout << "The data type of run_option is int, e.g. 0: run with paddle "
                "inference on cpu; \n"
            << std::endl;

    return app_options;
  }

  app_options.inference_backend = BACKENDS[ std::atoi(argv[5]) ];
  
  app_options.detection_model_dir = argv[1];
  app_options.classification_model_dir = argv[2];
  app_options.recognition_model_dir = argv[3];
  app_options.recognition_label_file_dir = argv[4];
  
  if ( argc >= 7 ) {
    app_options.server_port = std::stoi( argv[6] );
  }

  return app_options;
}


Settings createSettings( AppOptions app_options ) {

  std::cout << "\n det_model_dir: " << app_options.detection_model_dir <<
               "\n cls_model_dir: " << app_options.classification_model_dir <<
               "\n rec_model_dir: " << app_options.recognition_model_dir <<
               "\n rec_label_file: " << app_options.recognition_label_file_dir <<
               "\n inference_backend: " << app_options.inference_backend <<
               "\n server_port: " << app_options.server_port << "\n"
            << std::endl;

  Settings settings;
  fastdeploy::RuntimeOption runtime_option;

  std::string backend = app_options.inference_backend;

  if ( backend == "Paddle_CPU" ) {
    runtime_option.UseCpu();    
    runtime_option.UsePaddleBackend(); // Paddle Inference | 0
  }
  else if ( backend == "Open_VINO" ) {
    runtime_option.UseCpu();
    runtime_option.UseOpenVINOBackend(); // OpenVINO | 1
  }
  else if ( backend == "ONNX_CPU" ) {
    runtime_option.UseCpu();
    runtime_option.UseOrtBackend(); // ONNX Runtime | 2
  }
  else if ( backend == "Paddle_Lite" ) {
    runtime_option.UseCpu();    
    runtime_option.UseLiteBackend(); // Paddle Lite | 3
  }
  else if ( backend == "Paddle_GPU" ) {
    runtime_option.UseGpu();
    runtime_option.UsePaddleBackend(); // Paddle Inference | 4
  }
  else if ( backend == "Paddle_GPU_Tensor_RT" ) {
    runtime_option.UseGpu();
    runtime_option.UsePaddleInferBackend();
    runtime_option.paddle_infer_option.collect_trt_shape = true;
    runtime_option.paddle_infer_option.enable_trt = true; // Paddle-TensorRT | 5
  }
  else if ( backend == "ONNX_GPU" ) {
    runtime_option.UseGpu();
    runtime_option.UseOrtBackend(); // ONNX Runtime | 6
  }
  else if ( backend == "Tensor_RT" ) {
    runtime_option.UseGpu();
    runtime_option.UseTrtBackend(); // TensorRT | 7
  }  

  settings.models = getModels(
    app_options.detection_model_dir,
    app_options.classification_model_dir,
    app_options.recognition_model_dir,
    app_options.recognition_label_file_dir,
    runtime_option
  );

  settings.server_port = app_options.server_port;

  return settings;
}

void updateSettings( Settings& settings, AppOptions& new_app_options ) {
  std::cout << "\n Updating settings... \n" << std::endl;
  settings = createSettings( new_app_options );
}