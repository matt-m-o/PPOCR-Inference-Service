#ifndef INFERENCE_PIPELINE_BUILDER_HPP
#define INFERENCE_PIPELINE_BUILDER_HPP

#include <fastdeploy/vision.h>
#include "inference_models_manager.hpp"
#include "util.hpp"

int const cls_batch_size = 1;
int const rec_batch_size = 6;


class InferencePipelineBuilder {

private:
    InferenceModelsManager inference_models_manager;
    // std::map<std::string, std::vector<int64_t>> shape_info;

public:
    InferencePipelineBuilder() = default;

    std::shared_ptr< fastdeploy::pipeline::PPOCRv4 > buildInferencePipeline(    
        const std::string &det_model_dir,
        const std::string &cls_model_dir,
        const std::string &rec_model_dir,
        const std::string &rec_label_file,
        const AppSettingsPreset &app_settings
    ) {

        const std::string models_dir = "./models/";
        const std::string recognition_label_files_dir = "./recognition_label_files/";

        Models models = inference_models_manager.getOCRModels(
            models_dir + det_model_dir,
            models_dir + cls_model_dir,
            models_dir + rec_model_dir,
            recognition_label_files_dir + rec_label_file,
            app_settings
        );

        // auto detection_model = inference_models_manager.loadDetectionModel( models_dir + det_model_dir, runtime_option );
        // auto classification_model = inference_models_manager.loadClassificationModel( models_dir + cls_model_dir, runtime_option );
        // auto recognition_model = inference_models_manager.loadRecognitionModel(
        //     models_dir + rec_model_dir,
        //     recognition_label_files_dir + rec_label_file,
        //     runtime_option
        // );

        // auto _test = &(*detection_model);

        // The classification model is optional, so the PP-OCR can also be connected
        // in series as follows
        // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
        // auto ppocr_v4 =
        //     fastdeploy::pipeline::PPOCRv4(&det_model, &cls_model, &rec_model);
        auto pipeline = std::make_shared< fastdeploy::pipeline::PPOCRv4 >(
            // &( *(detection_model.get()) ),
            // &( *(classification_model.get()) ),
            // &( *(recognition_model.get()) )

            // detection_model,
            // classification_model,
            // recognition_model

            models.detection_model,
            models.classification_model,
            models.recognition_model


        );        
        
        // Set inference batch size for cls model and rec model, the value could be -1
        // and 1 to positive infinity.
        // When inference batch size is set to -1, it means that the inference batch
        // size
        // of the cls and rec models will be the same as the number of boxes detected
        // by the det model.
        pipeline->SetClsBatchSize(cls_batch_size);
        pipeline->SetRecBatchSize(rec_batch_size);

        // pipeline2.SetClsBatchSize(cls_batch_size);
        // pipeline2.SetRecBatchSize(rec_batch_size);

        // std::cout << "Initialized: " << pipeline->Initialized() << std::endl;
        // std::cout << "Initialized: " << pipeline2.Initialized() << std::endl;

        if (!pipeline->Initialized()) {
            std::cerr << "Failed to initialize PP-OCR." << std::endl;
            // return;
        }

        return pipeline;
    }
};






#endif