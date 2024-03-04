#ifndef INFERENCE_MANAGER_HPP
#define INFERENCE_MANAGER_HPP

// #include "settings.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fastdeploy/vision.h>
using json = nlohmann::json;

#include "../../includes/cpp-base64-2.rc.08/base64.cpp"
#include "settings_manager.hpp"
#include "inference_pipeline_builder.hpp"
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
        InferencePipelineBuilder pipeline_builder;
        std::unordered_map< std::string, std::shared_ptr<fastdeploy::pipeline::PPOCRv4> > pipelines;

        std::map< std::string, LanguagePreset > language_presets;
        AppSettingsPreset app_settings;

    public:
        InferenceManager() = default;

        void init(
            const std::map< std::string, LanguagePreset > language_presets,
            const AppSettingsPreset app_settings
        ) {
            this->language_presets = language_presets;
            this->app_settings = app_settings;
        }

        // Initialize one pipeline for each available language preset (uses more RAM)
        void initAll(
            const std::map< std::string, LanguagePreset > language_presets,
            const AppSettingsPreset app_settings
        ) {

            init(
                language_presets,
                this->app_settings
            );

            for ( const auto& pair : language_presets ) {                

                initPipeline( pair.first );
            }
        }

        void initPipeline( const std::string language_code ) {

            auto pipeline_it = pipelines.find( language_code );

            if ( pipeline_it != pipelines.end() ) {
                std::cout << "Pipeline for [" << language_code << "] already exists!" << std::endl;
                return;
            }

            auto language_preset_it = this->language_presets.find( language_code );

            if ( language_preset_it == language_presets.end() ) {
                std::cout << "Language preset for [" << language_code << "] does not exists!" << std::endl;
                return;
            }
            
            auto language_preset = language_preset_it->second;

            auto new_pipeline = pipeline_builder.buildInferencePipeline(
                language_preset.detection_model_dir,
                language_preset.classification_model_dir,
                language_preset.recognition_model_dir,
                language_preset.recognition_label_file_dir,
                app_settings
            );

            pipelines[ language_preset.language_code ] = new_pipeline;
        }        

        std::shared_ptr< fastdeploy::pipeline::PPOCRv4 > getPipeline( std::string language_code ) {

            initPipeline( language_code );

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

            infer_result.context_resolution = context_resolution;

            if ( result.boxes.empty() ) {
                return infer_result;
            }

            infer_result.ocr_result = result;

            // ocr_pipeline->ReleaseReusedBuffer();

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
                return infer( image, language_code );
            } else {
                std::cerr << "Failed to load the image." << std::endl;
            }

            return result;
        }

        InferenceResult inferBufferString( const std::string& image_str, std::string language_code ) {

            InferenceResult result;

            std::vector<uchar> data(image_str.begin(), image_str.end());
            cv::Mat image = cv::imdecode( data, cv::IMREAD_COLOR );
            

            if ( !image.empty() ) {
                // Image loaded successfully
                // cv::imshow("Loaded Image", image);
                // cv::waitKey(0);
                return infer( image, language_code );
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

      boxVertexIdx++; // [ 0 ... 3 ]
      bAxisIdx++; // [ 0 ... 7]
    }


    resultsJson["results"].push_back(itemJson);
    itemIdx++;
  }

  return resultsJson;
}

#endif // PPOCR_INFER
