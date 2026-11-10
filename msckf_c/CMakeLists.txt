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

option(SHUR_USE_LAPACK "Use LAPACK for dense camera Cholesky" OFF)
if(SHUR_USE_LAPACK)
  find_package(LAPACK REQUIRED)
  target_link_libraries(shur_runner PRIVATE ${LAPACK_LIBRARIES})
  target_compile_definitions(shur_runner PRIVATE SHUR_USE_LAPACK=1)
endif()

# Opt in only when the binary will run on this build host or a compatible CPU.
option(SHUR_NATIVE_ARCH "Optimize shur_runner for the build host CPU" OFF)
if(SHUR_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" SHUR_HAS_MARCH_NATIVE)
  if(NOT SHUR_HAS_MARCH_NATIVE)
    message(FATAL_ERROR "SHUR_NATIVE_ARCH requires compiler support for -march=native")
  endif()
  target_compile_options(shur_runner PRIVATE -march=native)
endif()

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
