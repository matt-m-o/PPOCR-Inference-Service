#include "../hpp/inference_manager.hpp"
#include "../hpp/settings_manager.hpp"
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


int main( int argc, char *argv[] ) {  

  // OCR 

  std::cout << "Notice: this application is experimental!!! \n" << std::endl;

  AppOptions app_options = handleAppArgs( argc, argv );

  SettingsManager settings_manager( app_options );

  InferenceManager inference_manager;

  inference_manager.initAll(
    settings_manager.language_presets,
    settings_manager.getAppSettingsPreset()
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

    std::string id = bodyJson["id"].get<std::string>();
    std::string language_code = bodyJson["language_code"].get<std::string>();
    std::string base64EncodedImage = bodyJson["base64Image"].get<std::string>();

    // std::cout << "\n Request language_code: " << language_code << std::endl;

    InferenceResult const result = inference_manager.inferBase64( base64EncodedImage, language_code );

    auto resultJson = ocrResultToJson( result );
    resultJson["id"] = id;

    res.set_content( resultJson.dump(), "application/json" );
  });

  svr.Get("/supported-languages", [&](const Request & /*req*/, Response &res) {

    nlohmann::json resultsJson;
    resultsJson["language_codes"] = json::array();

    for ( const std::string& language_code : settings_manager.getAvailableLanguages() ) {
      
      resultsJson["language_codes"].push_back( language_code );      
    }

    res.set_content( resultsJson.dump(), "application/json" );
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
  
  std::cout << "\n Listening on port: " << settings_manager.getServerPort() << std::endl;
  svr.listen( "0.0.0.0", settings_manager.getServerPort() );
  
  return 0;
}
