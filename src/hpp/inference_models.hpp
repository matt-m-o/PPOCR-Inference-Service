#ifndef INFERENCE_MODELS_HPP
#define INFERENCE_MODELS_HPP

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

public:
    InferenceModelsManager() = default;

    Models getOCRModels(
        const std::string &det_model_dir,
        const std::string &cls_model_dir,
        const std::string &rec_model_dir,
        const std::string &rec_label_file,
        const fastdeploy::RuntimeOption &runtime_option
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
        models.detection_model = loadDetectionModel( det_model_dir, runtime_option );
        models.classification_model = loadClassificationModel( cls_model_dir, runtime_option );
        models.recognition_model = loadRecognitionModel( rec_model_dir, rec_label_file, runtime_option );

        return models;
    }

    fastdeploy::vision::ocr::DBDetector* loadDetectionModel(
        const std::string &det_model_dir,
        const fastdeploy::RuntimeOption &runtime_option
    ) {

        auto model = detection_models.find( det_model_dir );

        if ( model != detection_models.end() ) {
            return model->second.get(); // Getting variable from pointer
        }

        std::cout <<"\n loadDetectionModel \n"<< std::endl;
        
        auto det_model_file = det_model_dir + sep + "inference.pdmodel";
        auto det_params_file = det_model_dir + sep + "inference.pdiparams";

        detection_models[det_model_dir] = std::make_shared< fastdeploy::vision::ocr::DBDetector >(
        det_model_file, det_params_file, runtime_option);
        

        assert( detection_models[det_model_dir]->Initialized() );

        detection_models[det_model_dir]->GetPreprocessor().SetMaxSideLen(960);
        detection_models[det_model_dir]->GetPostprocessor().SetDetDBThresh(0.3);
        detection_models[det_model_dir]->GetPostprocessor().SetDetDBBoxThresh(0.6);
        detection_models[det_model_dir]->GetPostprocessor().SetDetDBUnclipRatio(1.5);
        detection_models[det_model_dir]->GetPostprocessor().SetDetDBScoreMode("slow");
        detection_models[det_model_dir]->GetPostprocessor().SetUseDilation(0);
        

        return detection_models[det_model_dir].get();
    }

    fastdeploy::vision::ocr::Classifier* loadClassificationModel(
        const std::string &cls_model_dir,
        const fastdeploy::RuntimeOption &runtime_option
    ) {

        auto model = classification_models.find( cls_model_dir );

        if ( model != classification_models.end() ) {
            return model->second.get();
        }

        auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
        auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

        classification_models[cls_model_dir] = std::make_shared< fastdeploy::vision::ocr::Classifier >(
        cls_model_file, cls_params_file, runtime_option);        

        assert( classification_models[cls_model_dir]->Initialized() );

        classification_models[cls_model_dir]->GetPostprocessor().SetClsThresh(0.9);

        // classification_models[cls_model_dir] = cls_model;

        return classification_models[cls_model_dir].get();
    }

    fastdeploy::vision::ocr::Recognizer* loadRecognitionModel(
        const std::string &rec_model_dir,
        const std::string &rec_label_file,
        const fastdeploy::RuntimeOption &runtime_option
    ) {
        auto model = recognition_models.find( rec_model_dir );

        if ( model != recognition_models.end() ) {
            return model->second.get();
        }
        
        auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
        auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

        auto rec_option = runtime_option;

        recognition_models[rec_model_dir] = std::make_shared< fastdeploy::vision::ocr::Recognizer >(
        rec_model_file, rec_params_file, rec_label_file, rec_option);
        

        assert( recognition_models[rec_model_dir]->Initialized() );

        // recognition_models[rec_model_dir] = rec_model;

        return recognition_models[rec_model_dir].get();
    }
};

#endif