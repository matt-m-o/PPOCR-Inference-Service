#ifndef PPOCR_INFER_H
#define PPOCR_INFER_H

// #include "settings.hpp"
#include <iostream>
#include <fastdeploy/vision.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "../../includes/cpp-base64-2.rc.08/base64.cpp"
#include "settings.hpp"
#include "inference_pipeline.hpp"
#include "util.hpp"

struct ContextResolution {
  int width;
  int height;
};

struct InferenceResult {
  fastdeploy::vision::OCRResult ocr_result;
  ContextResolution context_resolution;
};

class InferenceManager {

    private:
        PipelineBuilder pipeline_builder;
        std::unordered_map< std::string, std::shared_ptr<fastdeploy::pipeline::PPOCRv4> > pipelines;

    public:
        InferenceManager() = default;

        // Initialize one pipeline for each available language preset (uses more RAM)
        void initAll( SettingsManager& settings_manager ) {

            for ( const auto& pair : settings_manager.language_presets ) {

                LanguagePreset language_preset = pair.second;

                auto pipeline = pipeline_builder.buildInferencePipeline(
                    language_preset.detection_model_dir,
                    language_preset.classification_model_dir,
                    language_preset.recognition_model_dir,
                    language_preset.recognition_label_file_dir,
                    settings_manager.getInferenceBackend()
                );

                // std::cout << "initAll. Initialized: " << pipeline->Initialized() << std::endl;

                pipelines[ language_preset.language_code ] = pipeline;
            }
        }

        // Initialize one pipeline for the current default language preset (uses less RAM)
        void initSingle( SettingsManager& settings_manager ) {

            LanguagePreset language_preset = settings_manager.language_presets[ settings_manager.getDefaultLanguageCode() ];
                                
            pipelines[ language_preset.language_code ] = pipeline_builder.buildInferencePipeline(
                language_preset.detection_model_dir,
                language_preset.classification_model_dir,
                language_preset.recognition_model_dir,
                language_preset.recognition_label_file_dir,
                settings_manager.getInferenceBackend()
            );        
        }

        std::shared_ptr< fastdeploy::pipeline::PPOCRv4 > getPipeline( std::string language_code ) {            

            auto it = pipelines.find( language_code );

            if ( it != pipelines.end() ) {

                std::shared_ptr< fastdeploy::pipeline::PPOCRv4 > objPtr = it->second; // Retrieve the shared_ptr                

                return objPtr;
            } else {
                std::cerr << language_code <<" not found in the map." << std::endl;
            }

            return pipelines[ language_code ];
        }

        InferenceResult infer( const cv::Mat& image, std::string language_code ) {

            auto ocr_pipeline = getPipeline( language_code );

            // Access properties and call functions                
            // std::cout << "infer. Initialized: " << ocr_pipeline->Initialized() << std::endl;

            InferenceResult infer_result;            
            

            fastdeploy::vision::OCRResult result;
            if ( !ocr_pipeline->Predict(image, &result) ) {
                std::cerr << "Failed to predict." << std::endl;
                return infer_result;
            }            

            // auto im_bak = im.clone();
            // auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
            // cv::imwrite("vis_result.jpg", vis_im);
            // std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
            
            ContextResolution context_resolution;
            context_resolution.width = image.cols;
            context_resolution.height = image.rows;

            infer_result.ocr_result = result;
            infer_result.context_resolution = context_resolution;

            return infer_result;
        }

        InferenceResult inferBase64( const std::string& base64EncodedImage, std::string language_code ) {

            InferenceResult result;

            std::string decoded_data = base64_decode(base64EncodedImage);

            std::string dec_jpg =  base64_decode(base64EncodedImage);
            std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
            cv::Mat image = cv::imdecode(cv::Mat(data), 1);
            

            if ( !image.empty() ) {
                // Image loaded successfully
                // cv::imshow("Loaded Image", image);
                // cv::waitKey(0);
                result = infer( image, language_code );
            } else {
                std::cerr << "Failed to load the image." << std::endl;
            }

            return result;
        }
};


nlohmann::json ocrResultToJson( const InferenceResult& infer_result ) {

  auto ocrResult = infer_result.ocr_result;
  auto context_resolution = infer_result.context_resolution;

  std::array< std::string, 4 > const boxVerticesIndices = { "top_left", "top_right", "bottom_right", "bottom_left" };

  nlohmann::json resultsJson;
  resultsJson["id"] = "";
  resultsJson["results"] = json::array();
  resultsJson["context_resolution"] = { {"width", context_resolution.width}, {"height", context_resolution.height} };

  int itemIdx = 0;
  for ( const std::string& text : ocrResult.text ) {

    nlohmann::json itemJson;
    itemJson["text"] = text;
    itemJson["box"] = json();
    itemJson["score"] = ocrResult.cls_scores[itemIdx];

    auto const box = ocrResult.boxes[ itemIdx ];
    int boxVertexIdx = 0;
    for ( int bAxisIdx = 0; bAxisIdx < 8; ++bAxisIdx ) {

      const int x = box[ bAxisIdx ];
      const int y = box[ bAxisIdx+1 ];

      const std::string boxVertexName = boxVerticesIndices[ boxVertexIdx ];
      itemJson["box"][boxVertexName] = { {"x", x}, { "y", y } };

      boxVertexIdx++; // [ 1 ... 4 ]
      bAxisIdx++; // [ 1 ... 8]
    }


    resultsJson["results"].push_back(itemJson);
    itemIdx++;
  }

  return resultsJson;
}

#endif // PPOCR_INFER
