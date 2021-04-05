cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(rife-ncnn-vulkan)

cmake_minimum_required(VERSION 3.2)

# cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE RelWithDebInfo)

# compilation database (completion for Linux)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
find_package(PNG)
find_package(ncnn REQUIRED)
find_package(Vulkan REQUIRED)
if(NOT WIN32)
  find_package(X11)
  find_package(Threads REQUIRED)
endif()

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -g -fexceptions")

# enable global link time optimization
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_supported_output)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(WARNING "IPO is not supported: ${ipo_supported_output}")
endif()

set(DATA_DIR "${PROJECT_SOURCE_DIR}/data")
set(MODELS_DIR "${PROJECT_SOURCE_DIR}/models")
configure_file("${PROJECT_SOURCE_DIR}/FrameInterpolation/Settings.h.in"
  "${PROJECT_SOURCE_DIR}/FrameInterpolation/Settings.h")
set(CMAKE_EXTRA_INCLUDE_FILES)

add_subdirectory(FrameInterpolation)
