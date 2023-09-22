#ifndef UTIL_HPP
#define UTIL_HPP

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;


#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

nlohmann::json readJsonFile( std::string const path ) {
  std::ifstream file( path );
  json json_data = json::parse(file);
  file.close();
  
  return json_data;
}

void printJsonData( json& data ) {  
  std::cout << "[INFO-JSON]:" << data.dump() << std::endl;
}

#endif