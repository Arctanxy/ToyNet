cmake_minimum_required(VERSION 3.10)

project(DNN)

include_directories(include)

file(GLOB SOURCES src/*.cpp)

MESSAGE(STATUS "SOURCES files: ${SOURCES}.")

add_executable(DNN test.cpp ${SOURCES})