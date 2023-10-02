#ifndef UTIL_HPP
#define UTIL_HPP

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

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

void writeJsonFile( ordered_json& data, std::string& path ) {

  std::ofstream o( path );
  o << std::setw(4) << data << std::endl;
}

void printJsonData( json& data ) {  
  std::cout << "[INFO-JSON]:" << data.dump() << std::endl;
}

#endif