cmake_minimum_required(VERSION 3.10)

project(ToyNet)

include_directories(include)

file(GLOB SOURCES src/*.cpp)

MESSAGE(STATUS "SOURCES files: ${SOURCES}.")

add_executable(ToyNet test.cpp ${SOURCES})
