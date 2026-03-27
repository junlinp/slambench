cmake_minimum_required(VERSION 3.10)
project(msckf_c LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_library(msckf_linalg linalg.c)

add_executable(msckf_runner msckf.c)
target_link_libraries(msckf_runner PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner PRIVATE MSCKF_USE_ROW_QR=1)

add_executable(shur_runner shur.cc)
target_link_libraries(shur_runner PRIVATE msckf_linalg m)

add_executable(msckf_runner_row msckf.c)
target_link_libraries(msckf_runner_row PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner_row PRIVATE MSCKF_USE_ROW_QR=1)

add_executable(msckf_runner_col msckf.c)
target_link_libraries(msckf_runner_col PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner_col PRIVATE MSCKF_USE_COL_QR=1)

add_executable(msckf_runner_rr msckf.c)
target_link_libraries(msckf_runner_rr PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner_rr PRIVATE MSCKF_USE_ROW_QR=1)

add_executable(msckf_runner_rc msckf.c)
target_link_libraries(msckf_runner_rc PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner_rc PRIVATE MSCKF_USE_COL_QR=1)

add_executable(msckf_runner_cr msckf.c)
target_link_libraries(msckf_runner_cr PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner_cr PRIVATE MSCKF_STORAGE_COL_MAJOR=1 MSCKF_USE_ROW_QR=1)

add_executable(msckf_runner_cc msckf.c)
target_link_libraries(msckf_runner_cc PRIVATE msckf_linalg m)
target_compile_definitions(msckf_runner_cc PRIVATE MSCKF_STORAGE_COL_MAJOR=1 MSCKF_USE_COL_QR=1)

find_library(OPENBLAS_LIB NAMES openblas)
if(OPENBLAS_LIB)
  add_executable(msckf_runner_openblas msckf.c)
  target_link_libraries(msckf_runner_openblas PRIVATE msckf_linalg m ${OPENBLAS_LIB})
  target_compile_definitions(msckf_runner_openblas PRIVATE MSCKF_STORAGE_COL_MAJOR=1 MSCKF_USE_LAPACK=1 MSCKF_USE_OPENBLAS=1)
endif()

add_executable(msckf_runner_sliding msckf_sliding.c linalg.c)
target_link_libraries(msckf_runner_sliding PRIVATE m)
