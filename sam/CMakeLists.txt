cmake_minimum_required(VERSION 3.10)
project(sam_solver LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(sam_runner sam.cc ../msckf_c/linalg.c)
target_include_directories(sam_runner PRIVATE ../msckf_c ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(sam_runner PRIVATE m)
