CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

INCLUDE(FindPCAP.cmake)

SET(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(CUDA)


FILE(GLOB SOURCE_FILES src/main.cpp src/capture/*.cpp src/capture/*.h src/inspection/*.cpp src/inspection/*.h
src/inspection/*.cu src/ThreadPool/*.cpp src/ThreadPool/*.h src/util/*.cpp src/util/*.h)


IF(PROFILE)
    ADD_DEFINITIONS("-pg")
    ADD_DEFINITIONS("-g -O0")
    ADD_DEFINITIONS(-DVERBOSE)
    SET(CUDA_NVCC_FLAGS -g -G --default-stream per-thread)
ENDIF(PROFILE)

IF(VERBOSE)
    ADD_DEFINITIONS(-DVERBOSE)
ENDIF(VERBOSE)

IF(DEBUG)
    ADD_DEFINITIONS("-g -O0")
    ADD_DEFINITIONS(-DDEBUG)
    ADD_DEFINITIONS(-DVERBOSE)
ENDIF(DEBUG)

CUDA_ADD_EXECUTABLE(GPUNIDS ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(GPUNIDS -pg -pthread  ${PCAP_LIBRARY})
