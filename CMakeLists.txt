# Copyright (c) 2022, Ildar Kashaev. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(IPL CUDA CXX)

find_package(CUDA REQUIRED)
find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

find_library(NPPIF_LIBRARY nppif ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPIG_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

# LIBRARY TYPE
option(BUILD_STATIC "Build IPL as static library" OFF)

set(LIBTYPE SHARED)
if(BUILD_STATIC)
  set(LIBTYPE STATIC)
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
include_directories("${PROJECT_SOURCE_DIR}/include")

add_library(IPL ${LIBTYPE} ipl/JpegDecoder.cc ipl/JpegEncoder.cc
                           ipl/Image.cc ipl/ImageResizer.cc)

target_link_libraries(IPL ${CUDA_LIBRARIES} nvjpeg ${NPPIF_LIBRARY} ${NPPC_LIBRARY} ${NPPIG_LIBRARY})
