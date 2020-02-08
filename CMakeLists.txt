cmake_minimum_required(VERSION 3.5)
project(algo_practice)

set(CMAKE_CXX_STANDARD 11)

# CMake Package Collection
set(OpenCV_STATIC ON)
set(OpenCV_DIR /Users/wilsonc/makedeps/osx/OpenCV/lib/cmake/opencv4)
set(eigen3_DIR /Users/wilsonc/makedeps/osx/eigen3/share/eigen3/cmake)
set(Eigen3_DIR /Users/wilsonc/makedeps/osx/eigen3/share/eigen3/cmake)
find_package(OpenCV CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
find_package(Eigen3 REQUIRED)

set(LIB_NAME "sgbm_practice")

set(LIB_SRC
  src/main.cpp
  src/sgbm.cpp
  src/sgbm.h
  src/sgbm_c.cpp
  src/precomp.hpp
  )

include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}/opencv2)

# Generate Exe Files
add_executable("${LIB_NAME}_test" ${LIB_SRC})
target_link_libraries("${LIB_NAME}_test" ${OpenCV_LIBS})
