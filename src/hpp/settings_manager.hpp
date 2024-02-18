#ifndef SETTINGS_MANAGER_HPP
#define SETTINGS_MANAGER_HPP

// #include <string>
// #include "ppocr_infer.hpp"
#include "util.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Main input for changing settings
struct AppOptions {
  std::string app_settings_preset_root = "./presets/";
  std::string app_settings_preset_name = "default";
  std::string language_code = "default";
  std::string inference_backend = "default"; //! "Paddle_CPU", "Open_VINO", "ONNX_CPU", "Paddle_Lite", "Paddle_GPU", "Paddle_GPU_Tensor_RT", "ONNX_GPU", "Tensor_RT"
  int server_port = 0;
};

// App settings preset JSON
struct AppSettingsPreset {
  std::string app_settings_preset_name = "default";  
  std::map< std::string, std::string > language_presets; // < language_code, preset_name >
  std::string language_code;
  bool initialize_all_language_presets = false;
  std::string inference_backend;
  int cpu_threads = 8;
  int server_port;
  int max_image_width = 1920; // Maximum image width
  double det_db_thresh = 0.3; // Only pixels with a score greater than this threshold will be considered as text pixels
  double det_db_box_thresh = 0.6; // When the average score of all pixels is greater than the threshold, the result will be considered as a text area
  double det_db_unclip_ratio = 1.5; // Expansion factor of the Vatti clipping algorithm, which is used to expand the text area
  std::string det_db_score_mode = "slow"; // DB detection result score calculation method
  bool use_dilation = false; // Whether to inflate the segmentation results to obtain better detection results
  double cls_thresh = 0.9; // Prediction threshold, when the model prediction result is 180 degrees, and the score is greater than the threshold, the final prediction result is considered to be 180 degrees and needs to be flipped
};

struct UpdateAppSettingsPresetInput {
  std::string inference_backend;
  int cpu_threads;
  int max_image_width;
  double det_db_thresh;
  double det_db_box_thresh;
  double det_db_unclip_ratio;
  std::string det_db_score_mode;
  bool use_dilation = false;
  double cls_thresh = 0.9;
};


// Language preset JSON
struct LanguagePreset {
  std::string name;
  std::string language_code; // e.g. en, ja, ch
  std::string detection_model_dir;
  std::string classification_model_dir;
  std::string recognition_model_dir;
  std::string recognition_label_file_dir; // Also known as Dictionary
};



AppOptions handleAppArgs( int argc, char *argv[] ) {  

  AppOptions app_options;  

  if (argc == 1) {
    std::cout << "Args usage: [app_settings_preset_root] [app_settings_preset] [server_port] \n"
                 "e.g.: default ja 12345 \n"
                //  "e.g default ch \n"
              << std::endl;

    return app_options;
  }

  if ( argc > 1 ) {
    app_options.app_settings_preset_root = argv[1];
  }
  if ( argc > 2 ) {
    app_options.app_settings_preset_name = argv[2];
  }
  if ( argc > 3 ) { 
    app_options.server_port = std::atoi( argv[3] );
  }

  return app_options;
}



class SettingsManager {

  private:
    AppSettingsPreset app_settings_preset;
    AppOptions app_options;

  public:
    std::map< std::string, LanguagePreset > language_presets;

    SettingsManager() = default;

    SettingsManager( AppOptions app_options ) {
      this->app_options = app_options;
      initSettings();
    }

    void initSettings() {

      loadAppSettingsPreset();

      loadLanguagePresets();
    }

    std::string getInferenceBackend() {
      return app_settings_preset.inference_backend;
    }

    std::string getDefaultLanguageCode() {
      return app_settings_preset.language_code;
    }

    int getServerPort() {
      return app_settings_preset.server_port;
    }
    int getCpuThreads() {
      return app_settings_preset.cpu_threads;
    }
    int getMaxImageWidth() {
      return app_settings_preset.max_image_width;
    }

    void loadAppSettingsPreset() {

      std::cout <<"\n App settings preset root: " << app_options.app_settings_preset_root << std::endl;
      
      app_settings_preset.app_settings_preset_name = app_options.app_settings_preset_name;

      if ( app_options.app_settings_preset_root == "default" ) {
        app_options.app_settings_preset_root = "./presets/";
      }
      
      
      json app_settings_preset_json = readJsonFile(
        app_options.app_settings_preset_root + app_options.app_settings_preset_name + ".json"
      );

      app_settings_preset.inference_backend = app_settings_preset_json["inference_backend"].get< std::string >();
      app_settings_preset.server_port = app_settings_preset_json["port"].get<int>();
      app_settings_preset.language_code = app_settings_preset_json["language_code"].get< std::string >();
      app_settings_preset.cpu_threads = app_settings_preset_json["cpu_threads"].get< int >();
      app_settings_preset.max_image_width = app_settings_preset_json["max_image_width"].get< int >();
      app_settings_preset.det_db_thresh = app_settings_preset_json["det_db_thresh"].get< double >();
      app_settings_preset.det_db_box_thresh = app_settings_preset_json["det_db_box_thresh"].get< double >();
      app_settings_preset.det_db_unclip_ratio = app_settings_preset_json["det_db_unclip_ratio"].get< double >();
      app_settings_preset.det_db_score_mode = app_settings_preset_json["det_db_score_mode"].get< std::string >();
      app_settings_preset.use_dilation = app_settings_preset_json["use_dilation"].get< bool >();
      app_settings_preset.cls_thresh = app_settings_preset_json["cls_thresh"].get< double >();

      if ( app_settings_preset_json["language_presets"].is_null() )
        return;

      for ( auto& el : app_settings_preset_json["language_presets"].items() ) {        
        app_settings_preset.language_presets[ el.key() ] = el.value().get<std::string>();
      }

      // Overwriting preset options by args options
      if ( app_options.language_code != "default" ) {
        app_settings_preset.language_code = app_options.language_code;
      }
      if ( app_options.inference_backend != "default" ) {
        app_settings_preset.inference_backend = app_options.inference_backend;
      }
      if ( app_options.server_port != 0 ) {
        app_settings_preset.server_port = app_options.server_port;
      }

      std::cout <<"\n App settings preset: " << app_settings_preset.app_settings_preset_name << std::endl;
      std::cout << std::setw(4) << app_settings_preset_json << std::endl; 
    }

    // Loads the language presets from file according to selected AppSettingsPreset
    void loadLanguagePresets() {
      
      for (const auto& pair : app_settings_preset.language_presets) {
        
        LanguagePreset language_preset;
        language_preset.name =  pair.second;
        language_preset.language_code = pair.first;

        json language_preset_json = readJsonFile( "./presets/language_presets/" + language_preset.name + ".json" );


        language_preset.detection_model_dir = language_preset_json["detection_model_dir"].get<std::string>();
        language_preset.classification_model_dir = language_preset_json["classification_model_dir"].get<std::string>();
        language_preset.recognition_model_dir = language_preset_json["recognition_model_dir"].get<std::string>();
        language_preset.recognition_label_file_dir = language_preset_json["recognition_label_file_dir"].get<std::string>();

        language_presets[ language_preset.language_code ] = language_preset;

        std::cout << "language_code: " << language_preset.language_code << ", preset_name: " << language_preset.name << std::endl;
      }      
    }
    
    std::vector< std::string > getAvailableLanguages() {

      std::vector< std::string > languages_vector;

      for (const auto& pair : language_presets) {
        languages_vector.push_back(pair.first);        
      }

      return languages_vector;
    }

    AppSettingsPreset getAppSettingsPreset() {
      return app_settings_preset;
    }

    void updateSettingsPreset( UpdateAppSettingsPresetInput input ) {
      app_settings_preset.inference_backend = input.inference_backend;
      app_settings_preset.cpu_threads = input.cpu_threads;
      app_settings_preset.max_image_width = input.max_image_width;
      app_settings_preset.det_db_thresh = input.det_db_thresh;
      app_settings_preset.det_db_box_thresh = input.det_db_box_thresh;
      app_settings_preset.det_db_unclip_ratio = input.det_db_unclip_ratio;
      app_settings_preset.det_db_score_mode = input.det_db_score_mode;
      app_settings_preset.use_dilation = input.use_dilation;
      app_settings_preset.cls_thresh = input.cls_thresh;
    }

    void saveAppSettingsPreset() {

      std::string file_path = app_options.app_settings_preset_root;
      std::string file_name = app_settings_preset.app_settings_preset_name + ".json";

      nlohmann::ordered_json settings_preset_json;
      settings_preset_json["name"] = app_settings_preset.app_settings_preset_name;
      settings_preset_json["language_presets"] = json();
            
      for (const auto& pair : app_settings_preset.language_presets) {        
        settings_preset_json["language_presets"][pair.first] = pair.second;
      }

      settings_preset_json["language_code"] = app_settings_preset.language_code;
      settings_preset_json["initialize_all_language_presets"] = app_settings_preset.initialize_all_language_presets;
      settings_preset_json["inference_backend"] = app_settings_preset.inference_backend;
      settings_preset_json["cpu_threads"] = app_settings_preset.cpu_threads;
      settings_preset_json["port"] = app_settings_preset.server_port;
      settings_preset_json["max_image_width"] = app_settings_preset.max_image_width;
      settings_preset_json["det_db_thresh"] = app_settings_preset.det_db_thresh;
      settings_preset_json["det_db_box_thresh"] = app_settings_preset.det_db_box_thresh;
      settings_preset_json["det_db_unclip_ratio"] = app_settings_preset.det_db_unclip_ratio;
      settings_preset_json["det_db_score_mode"] = app_settings_preset.det_db_score_mode;
      settings_preset_json["use_dilation"] = app_settings_preset.use_dilation;
      settings_preset_json["cls_thresh"] = app_settings_preset.cls_thresh;
      
      file_path = file_path + file_name;
      std::cout << "Saving settings..." << std::endl;
      std::cout << "Settings file path: " << file_path << std::endl;
      std::cout << std::setw(4) << settings_preset_json << std::endl;      

      writeJsonFile( settings_preset_json, file_path );
    }
};



#endif