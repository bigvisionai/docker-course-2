# SPECIFY THE MINIMUM VERSION OF CMAKE REQUIRED
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# SPECIFY YOUR PROJECT NAME
PROJECT(sampleCode)

# FIND THE PACKAGES IN THE SYSTEM
find_package(OpenCV REQUIRED PATHS "/usr/local/lib/cmake/opencv4")
find_package(Leptonica REQUIRED)
find_package(Tesseract REQUIRED)
find_package(Torch REQUIRED PATHS "/usr/local/libtorch")
find_package(dlib REQUIRED PATHS "/usr/local/lib/cmake/dlib/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# INCLUDE PACKAGE DIRECTORIES
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Tesseract_INCLUDE_DIRS})
include_directories(${Leptonica_INCLUDE_DIRS})

add_executable(main main.cpp)

# LINK LIBRARIES TO THE EXECUTABLE FILE
target_link_libraries(main ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES} dlib::dlib)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
