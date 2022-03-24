cmake_minimum_required(VERSION 3.20)
project(TestTextureCUDA LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(TestTextureCUDA main.cpp shaderGray.cu shaderRGBA.cu shader.h cuda_helper.h)
