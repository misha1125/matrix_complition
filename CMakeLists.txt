cmake_minimum_required(VERSION 3.13)
project(complition)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover -llapack -lblas")

add_executable(complition main.cpp )