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


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using ppocr_service::RecognizeRequest;
using ppocr_service::RecognizeResponse;
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
        settings_manager.getInferenceBackend()
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

    Status Recognize(
      ServerContext* context,
      const RecognizeRequest* request,
      RecognizeResponse* response
    ) override {    

      InferenceResult const inference_result = inference_manager.inferBase64(
        request->base64_image(),
        request->language_code()
      );

      response->set_id( request->id() );

      auto context_resolution = response->mutable_context_resolution();
      context_resolution->set_width( inference_result.context_resolution.width );
      context_resolution->set_height( inference_result.context_resolution.height );
      
      int item_idx = 0;
      for ( const std::string& text : inference_result.ocr_result.text ) {

        auto new_result = response->add_results();
        new_result->set_text(text);
        new_result->set_score( inference_result.ocr_result.cls_scores[ item_idx ] );

        auto new_box = new_result->mutable_box();

        auto const box = inference_result.ocr_result.boxes[ item_idx ];
        int box_vertex_idx = 0;
        for ( int b_axis_idx = 0; b_axis_idx < 8; ++b_axis_idx ) {
        
          const int x = box[ b_axis_idx ];
          const int y = box[ b_axis_idx + 1 ];

          ppocr_service::Vertex* vertex;

          if ( box_vertex_idx == 0 ) {
            vertex = new_box->mutable_top_left();
          }
          else if ( box_vertex_idx == 1 ) {
            vertex = new_box->mutable_top_right();
          }
          else if ( box_vertex_idx == 2 ) {
            vertex = new_box->mutable_bottom_right();
          }
          else if ( box_vertex_idx == 3 ) {
            vertex = new_box->mutable_bottom_left();
          }

          vertex->set_x(x);
          vertex->set_y(y);

          box_vertex_idx++; // [ 1 ... 4 ]
          b_axis_idx++; // [ 1 ... 8]
        }

        item_idx++;
      }

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
