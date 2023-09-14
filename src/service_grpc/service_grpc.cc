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
#include "ppocr_service.grpc.pb.h"
#include "grpc_helpers.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using ppocr_service::RecognizeBase64Request;
using ppocr_service::RecognizeGenericResponse;
using ppocr_service::RecognizeBytesRequest;
using ppocr_service::SupportedLanguagesRequest;
using ppocr_service::SupportedLanguagesResponse;
using ppocr_service::PPOCRInference;


class PPOCRService final : public PPOCRInference::Service {

  private:
    SettingsManager settings_manager;
    InferenceManager inference_manager;
  
  public:

    PPOCRService( AppOptions app_options ) {

      settings_manager.initSettings( app_options );
      inference_manager.initAll(
        settings_manager.language_presets,
        settings_manager.getInferenceBackend(),
        settings_manager.getCpuThreads()
      );
    }

    SettingsManager& getSettingsManager() {
      return settings_manager;
    }

    int getServerPort() {
      return settings_manager.getServerPort();
    }

    Status SupportedLanguages(
      ServerContext* context,
      const SupportedLanguagesRequest* request,
      SupportedLanguagesResponse* response
    ) override {

      for ( const std::string& language_code : settings_manager.getAvailableLanguages() ) {
        
        response->add_language_codes(language_code);        
      }
      
      return Status::OK;
    }

    Status RecognizeBase64(
      ServerContext* context,
      const RecognizeBase64Request* request,
      RecognizeGenericResponse* response
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
      RecognizeGenericResponse* response
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
};


void RunServer( AppOptions& app_options ) {

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
  // Finally assemble the server.
  std::unique_ptr<Server> server( builder.BuildAndStart() );
  std::cout << "Server listening on " << server_address << std::endl;

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
