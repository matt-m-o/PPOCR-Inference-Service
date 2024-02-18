#include "../hpp/inference_manager.hpp"
#include "../hpp/settings_manager.hpp"
#include <chrono>
#include <cstdio>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include "ocr_service.grpc.pb.h"
#include "grpc_helpers.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using ocr_service::RecognizeBase64Request;
using ocr_service::RecognizeDefaultResponse;
using ocr_service::RecognizeBytesRequest;

using ocr_service::GetSupportedLanguagesRequest;
using ocr_service::GetSupportedLanguagesResponse;

using ocr_service::UpdatePpOcrSettingsRequest;
using ocr_service::UpdateSettingsResponse;


using ocr_service::OCRService;


class PPOCRService final : public OCRService::Service {

  private:
    SettingsManager settings_manager;
    InferenceManager inference_manager;
  
  public:

    PPOCRService( AppOptions app_options ) {

      settings_manager = SettingsManager( app_options );

      settings_manager.initSettings();
      inference_manager.init(
        settings_manager.language_presets,
        settings_manager.getAppSettingsPreset()
      );
    }

    SettingsManager& getSettingsManager() {
      return settings_manager;
    }

    int getServerPort() {
      return settings_manager.getServerPort();
    }

    Status GetSupportedLanguages(
      ServerContext* context,
      const GetSupportedLanguagesRequest* request,
      GetSupportedLanguagesResponse* response
    ) override {

      for ( const std::string& language_code : settings_manager.getAvailableLanguages() ) {
        
        response->add_language_codes(language_code);        
      }
      
      return Status::OK;
    }

    Status RecognizeBase64(
      ServerContext* context,
      const RecognizeBase64Request* request,
      RecognizeDefaultResponse* response
    ) override {    

      InferenceResult const inference_result = inference_manager.inferBase64(
        request->base64_image(),
        request->language_code()
      );

      response->set_id( request->id() );

      inferenceResultGRPCHelper( inference_result, response );

      return Status::OK;
    }

    Status RecognizeBytes(
      ServerContext* context,
      const RecognizeBytesRequest* request,
      RecognizeDefaultResponse* response
    ) override {    

      std::string image_str = request->image_bytes();
      
      InferenceResult const inference_result = inference_manager.inferBufferString(
        request->image_bytes(),
        request->language_code()
      );

      response->set_id( request->id() );

      inferenceResultGRPCHelper( inference_result, response );
      
      return Status::OK;
    }

    Status UpdatePpOcrSettings(
      ServerContext* context,
      const UpdatePpOcrSettingsRequest* request,
      UpdateSettingsResponse* response
    ) override {

      UpdateAppSettingsPresetInput settingsUpdate;
      settingsUpdate.inference_backend = request->inference_runtime();
      settingsUpdate.cpu_threads = request->cpu_threads();
      settingsUpdate.max_image_width = request->max_image_width();
      settingsUpdate.det_db_thresh = request->det_db_thresh();
      settingsUpdate.det_db_box_thresh = request->det_db_box_thresh();
      settingsUpdate.det_db_unclip_ratio = request->det_db_unclip_ratio();
      settingsUpdate.det_db_score_mode = request->det_db_score_mode();
      settingsUpdate.use_dilation = request->use_dilation();
      settingsUpdate.cls_thresh = request->cls_thresh();
      
      settings_manager.updateSettingsPreset( settingsUpdate );
      settings_manager.saveAppSettingsPreset();

      response->set_success( true );

      return Status::OK;
    }
};


void RunServer( AppOptions app_options ) {

  PPOCRService service( app_options );

  std::string server_address( "0.0.0.0:" + std::to_string( service.getServerPort() ) );

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort( server_address, grpc::InsecureServerCredentials() );

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize( 15 * 1024 * 1024 );

  // Finally assemble the server.
  std::unique_ptr<Server> server( builder.BuildAndStart() );

  nlohmann::json server_address_json = { { "server_address", server_address } };    
  printJsonData( server_address_json );

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}


int main( int argc, char *argv[] ) {  

  std::cout << "Notice: this application is experimental!!! \n" << std::endl;

  // OCR 
  AppOptions app_options = handleAppArgs( argc, argv );

  RunServer( app_options );
  
  return 0;
}
