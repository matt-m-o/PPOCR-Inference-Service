#include "hpp/settings.hpp";
#include "hpp/ppocr_infer.hpp"
#include <chrono>
#include <cstdio>
#include <httplib.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define SERVER_CERT_FILE "./cert.pem"
#define SERVER_PRIVATE_KEY_FILE "./key.pem"

using namespace httplib;



std::string dump_headers(const Headers &headers) {
  std::string s;
  char buf[BUFSIZ];

  for (auto it = headers.begin(); it != headers.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
    s += buf;
  }

  return s;
}

std::string log(const Request &req, const Response &res) {
  std::string s;
  char buf[BUFSIZ];

  s += "================================\n";

  snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
           req.version.c_str(), req.path.c_str());
  s += buf;

  std::string query;
  for (auto it = req.params.begin(); it != req.params.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%c%s=%s",
             (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
             x.second.c_str());
    query += buf;
  }
  snprintf(buf, sizeof(buf), "%s\n", query.c_str());
  s += buf;

  s += dump_headers(req.headers);

  s += "--------------------------------\n";

  snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
  s += buf;
  s += dump_headers(res.headers);
  s += "\n";

  if (!res.body.empty()) { s += res.body; }

  s += "\n";

  return s;
}



void reinitializePipeline( fastdeploy::pipeline::PPOCRv4& pipeline, Settings& settings ) {
  pipeline = initPipeline(
    settings.models.detectionModel,
    settings.models.classificationModel,
    settings.models.recognitionModel
    // settings
  );
}

int main( int argc, char *argv[] ) {  

  // OCR 

  AppOptions app_options = handleAppArgs( argc, argv );

  SettingsManager settings_manager( app_options );

  Settings settings = settings_manager.getSettings();
  
  fastdeploy::pipeline::PPOCRv4 ppocr_v4_pipeline = initPipeline(
    settings.models.detectionModel,
    settings.models.classificationModel,
    settings.models.recognitionModel
    // settings
  );


  // SERVER

  #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    SSLServer svr(SERVER_CERT_FILE, SERVER_PRIVATE_KEY_FILE);
  #else
    Server svr;
  #endif


  if (!svr.is_valid()) {
    printf("server has an error...\n");
    return -1;
  }

  svr.Post("/recognize", [&](const Request &req, Response &res,
                                    const ContentReader &content_reader) {
    std::string body;
    content_reader( [&]( const char *data, size_t data_length ) {
      body.append(data, data_length);
      return true;
    });

    auto bodyJson = json::parse(body);

    std::string base64EncodedImage = bodyJson["base64Image"].get<std::string>();      

    InferResult const result = inferBase64( base64EncodedImage, ppocr_v4_pipeline );

    auto resultJson = ocrResultToJson( result );

    res.set_content( resultJson.dump(), "application/json" );
  });

  svr.Post("/settings", [&]( const Request &req, Response &res,
                             const ContentReader &content_reader) {
    std::string body;
    content_reader( [&]( const char *data, size_t data_length ) {
      body.append(data, data_length);
      return true;
    });

    json bodyJson = json::parse(body);

    if ( bodyJson.empty() == false ) {            

      if ( bodyJson["app_settings_preset_name"].is_string() )
        app_options.app_settings_preset_name = bodyJson["app_settings_preset_name"];
      
      if ( bodyJson["language_code"].is_string() )
        app_options.language_code = bodyJson["language_code"];
      
      if ( bodyJson["inference_backend"].is_string() )
        app_options.inference_backend = bodyJson["inference_backend"];
      
      settings_manager.updateSettings( app_options );

      reinitializePipeline( ppocr_v4_pipeline, settings_manager.getSettings() );

      res.set_content( "", "application/json" );
    }

    
  });


  svr.set_error_handler([](const Request & /*req*/, Response &res) {
    const char *fmt = "<p>Error Status: <span style='color:red;'>%d</span></p>";
    char buf[BUFSIZ];
    snprintf(buf, sizeof(buf), fmt, res.status);
    res.set_content(buf, "text/html");
  });

  // svr.set_logger([](const Request &req, const Response &res) {
  //   printf("%s", log(req, res).c_str());
  // });
  
  std::cout << "\n Listening on port: " << settings.server_port << std::endl;
  svr.listen("0.0.0.0", settings.server_port);
  
  return 0;
}
