#ifndef INFERENCE_MODELS_MANAGER_HPP
#define INFERENCE_MODELS_MANAGER_HPP

#include <fastdeploy/vision.h>
#include "util.hpp"


struct Models {
    fastdeploy::vision::ocr::DBDetector* detection_model;
    fastdeploy::vision::ocr::Classifier* classification_model;
    fastdeploy::vision::ocr::Recognizer* recognition_model;
};


class InferenceModelsManager {

private:
    std::unordered_map< std::string, std::shared_ptr< fastdeploy::vision::ocr::DBDetector > > detection_models;
    std::unordered_map< std::string, std::shared_ptr< fastdeploy::vision::ocr::Classifier > > classification_models;
    std::unordered_map< std::string, std::shared_ptr< fastdeploy::vision::ocr::Recognizer > > recognition_models;

    fastdeploy::RuntimeOption det_runtime_option;
    fastdeploy::RuntimeOption cls_runtime_option;
    fastdeploy::RuntimeOption rec_runtime_option;

public:
    InferenceModelsManager() = default;

    Models getOCRModels(
        const std::string &det_model_dir,
        const std::string &cls_model_dir,
        const std::string &rec_model_dir,
        const std::string &rec_label_file,
        const AppSettingsPreset &app_settings
    ) {

        std::cout <<"\n  Models: " <<
                    "\n  det_model_dir: " << det_model_dir <<
                    "\n  cls_model_dir: " << cls_model_dir <<
                    "\n  rec_model_dir: " << rec_model_dir <<
                    "\n  rec_label_file: " << rec_label_file << "\n"
        << std::endl;

        // The cls and rec model can inference a batch of images now.
        // User could initialize the inference batch size and set them after create
        // PP-OCR model.
        /* int cls_batch_size = 1;
        int rec_batch_size = 6; */

        // If use TRT backend, the dynamic shape will be set as follow.
        // We recommend that users set the length and height of the detection model to
        // a multiple of 32.
        // We also recommend that users set the Trt input shape as follow.
        /* det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                                    {1, 3, 960, 960});
        cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
                                    {cls_batch_size, 3, 48, 1024});
        rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                                    {rec_batch_size, 3, 48, 2304}); */

        Models models;
        models.detection_model = loadDetectionModel( det_model_dir, app_settings );
        models.classification_model = loadClassificationModel( cls_model_dir, app_settings );
        models.recognition_model = loadRecognitionModel( rec_model_dir, rec_label_file, app_settings );

        return models;
    }

    fastdeploy::vision::ocr::DBDetector* loadDetectionModel(
        const std::string &det_model_dir,
        const AppSettingsPreset &app_settings
    ) {

        auto model = detection_models.find( det_model_dir );

        if ( model != detection_models.end() ) {
            return model->second.get(); // Getting variable from pointer
        }
        
        auto det_model_file = det_model_dir + sep + "inference.pdmodel";
        auto det_params_file = det_model_dir + sep + "inference.pdiparams";

        this->initRuntimeOption( det_runtime_option, app_settings );

        if ( app_settings.inference_backend == "Open_VINO" ) {
            det_runtime_option.openvino_option.SetShapeInfo( 
                {{ "x", {  1, 3, -1, -1 } }}
            );
        }

        detection_models[det_model_dir] = std::make_shared< fastdeploy::vision::ocr::DBDetector >(
            det_model_file, det_params_file, det_runtime_option
        );
        

        assert( detection_models[det_model_dir]->Initialized() );

        
        detection_models[det_model_dir]->GetPreprocessor()
            .SetMaxSideLen( app_settings.max_image_width );

        detection_models[det_model_dir]->GetPostprocessor()
            .SetDetDBThresh( app_settings.det_db_thresh );

        detection_models[det_model_dir]->GetPostprocessor()
            .SetDetDBBoxThresh( app_settings.det_db_box_thresh );

        detection_models[det_model_dir]->GetPostprocessor()
            .SetDetDBUnclipRatio( app_settings.det_db_unclip_ratio );

        detection_models[det_model_dir]->GetPostprocessor()
            .SetDetDBScoreMode( app_settings.det_db_score_mode );

        detection_models[det_model_dir]->GetPostprocessor()
            .SetUseDilation( app_settings.use_dilation );
        

        return detection_models[det_model_dir].get();
    }

    fastdeploy::vision::ocr::Classifier* loadClassificationModel(
        const std::string &cls_model_dir,
        const AppSettingsPreset &app_settings
    ) {

        auto model = classification_models.find( cls_model_dir );

        if ( model != classification_models.end() ) {
            return model->second.get();
        }

        auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
        auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

        this->initRuntimeOption( cls_runtime_option, app_settings );

        classification_models[cls_model_dir] = std::make_shared< fastdeploy::vision::ocr::Classifier >(
            cls_model_file, cls_params_file, cls_runtime_option
        );

        assert( classification_models[cls_model_dir]->Initialized() );

        classification_models[cls_model_dir]->GetPostprocessor()
            .SetClsThresh( app_settings.cls_thresh );

        return classification_models[cls_model_dir].get();
    }

    fastdeploy::vision::ocr::Recognizer* loadRecognitionModel(
        const std::string &rec_model_dir,
        const std::string &rec_label_file,
        const AppSettingsPreset &app_settings
    ) {
        auto model = recognition_models.find( rec_model_dir );

        if ( model != recognition_models.end() ) {
            return model->second.get();
        }
        
        auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
        auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";


        this->initRuntimeOption( rec_runtime_option, app_settings );

        recognition_models[rec_model_dir] = std::make_shared< fastdeploy::vision::ocr::Recognizer >(
            rec_model_file, rec_params_file, rec_label_file, rec_runtime_option
        );

        assert( recognition_models[rec_model_dir]->Initialized() );

        return recognition_models[rec_model_dir].get();
    }

    fastdeploy::RuntimeOption& initRuntimeOption(
        fastdeploy::RuntimeOption &runtime_option,
        const AppSettingsPreset &app_settings
    ) {
        auto const backend = app_settings.inference_backend;
        auto const cpu_threads = app_settings.cpu_threads;

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

        if ( cpu_threads > 0 && backend != "ONNX_CPU" ) { // Change cpu_threads while using ONNX can cause problems
            // std::cout << "SetCpuThreadNum: " << cpu_threads << std::endl;
            runtime_option.SetCpuThreadNum(cpu_threads);
        }

        return runtime_option;
    }
};

#endif