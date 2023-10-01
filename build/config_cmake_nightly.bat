@echo off
setlocal enabledelayedexpansion

set "current_directory=%CD%"
set "current_directory=!current_directory:\=/!"

@echo on
cmake .. -G "Visual Studio 17 2022" -A x64 -DFASTDEPLOY_INSTALL_DIR=%current_directory%/../sdk/fastdeploy-win-x64-0.0.0

@echo off
endlocal