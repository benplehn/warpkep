cmake_minimum_required(VERSION 3.24)

project(warpkep LANGUAGES CXX)

add_library(warpkep STATIC src/version.cpp)
target_include_directories(warpkep PUBLIC include)
target_compile_features(warpkep PUBLIC cxx_std_17)

add_executable(warpkep_smoke src/main.cpp)
target_link_libraries(warpkep_smoke PRIVATE warpkep)