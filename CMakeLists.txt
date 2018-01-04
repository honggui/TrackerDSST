cmake_minimum_required(VERSION 2.8)
project(test)

list(APPEND CMAKE_MODULE_PATH /usr/local/AID/opencv3.3.0/share/OpenCV)

find_package(OpenCV)


if(NOT WIN32)
ADD_DEFINITIONS("-std=c++11 -O3")
endif(NOT WIN32)

include_directories(${OpenCV_DIR}/include)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF ${sourcefiles} )
target_link_libraries( KCF ${OpenCV_LIBS})

SET(CMAKE_BUILD_TYPE "Debug")
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
ENDIF(CMAKE_COMPILER_IS_GNUCC)


