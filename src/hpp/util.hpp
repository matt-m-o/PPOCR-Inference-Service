#include <nlohmann/json.hpp>
using json = nlohmann::json;

nlohmann::json readJsonFile( std::string const path ) {
  std::ifstream file( path );
  json json_data = json::parse(file);
  file.close();
  
  return json_data;
}