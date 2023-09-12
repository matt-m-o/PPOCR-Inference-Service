#ifndef SETTINGS_HPP
#define SETTINGS_HPP

// #include <string>
// #include "ppocr_infer.hpp"
#include "util.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Main input for changing settings
struct AppOptions {
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
  int server_port;
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
    std::cout << "Args usage: [app_settings_preset] [language_code] [server_port] \n"
                 "e.g.: default ja 12345 \n"
                //  "e.g default ch \n"
              << std::endl;

    return app_options;
  }

  if ( argc > 1 ) {
    app_options.app_settings_preset_name = argv[1];
  }
  if ( argc > 2 ) {
    app_options.language_code = argv[2];
  }
  if ( argc > 3 ) {
    app_options.server_port = std::atoi( argv[3] );
  }

  return app_options;
}



class SettingsManager {

  private:
    AppSettingsPreset app_settings_preset;

  public:
    std::map< std::string, LanguagePreset > language_presets;

    SettingsManager( AppOptions app_options ) {
      initSettings( app_options );
    }

    void initSettings( AppOptions app_options ) {      

      loadAppSettingsPreset( app_options );

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

    void loadAppSettingsPreset( AppOptions& app_options ) {
      
      app_settings_preset.app_settings_preset_name = app_options.app_settings_preset_name;
      
      json app_settings_preset_json = readJsonFile( "./presets/" + app_options.app_settings_preset_name + ".json" );

      app_settings_preset.inference_backend = app_settings_preset_json["inference_backend"].get< std::string >();
      app_settings_preset.server_port = app_settings_preset_json["port"].get<int>();
      app_settings_preset.language_code = app_settings_preset_json["language_code"].get< std::string >();      

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

      std::cout <<"\n App settings: " << app_settings_preset.app_settings_preset_name <<                  
                  "\n  inference_backend: " << app_settings_preset.inference_backend <<
                  "\n  server_port: " << app_settings_preset.server_port << "\n"                 
      << std::endl;
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
    

    void updateSettings( AppOptions app_options ) {

      std::cout << "\n Updating settings... \n" << std::endl;      

      if ( app_options.app_settings_preset_name != app_settings_preset.app_settings_preset_name ) {
        loadAppSettingsPreset( app_options ); // Reload app settings preset
        loadLanguagePresets();
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

    }

    
    std::vector< std::string > getAvailableLanguages() {

      std::vector< std::string > languages_vector;

      for (const auto& pair : language_presets) {
        languages_vector.push_back(pair.first);        
      }

      return languages_vector;
    }

};



#endif