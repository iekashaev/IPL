// Copyright (c) 2022, Ildar Kashaev. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ipl/ImageResizer.h"

#include <stdexcept>

namespace ipl {

ImageResizer::ImageResizer() {}

ImageResizer::~ImageResizer() {}

bool ImageResizer::init() {
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamDefault) != cudaSuccess)
    return false;
  if (nppSetStream(stream_) != NPP_NO_ERROR) return false;

  npp_stream_ctx_.hStream = stream_;

  return true;
}

Image ImageResizer::resize(const Image& image,
                           std::initializer_list<int> size) {
  int dst_channels = image.get_size().at(0);

  NppiSize src_size;
  src_size.height = image.get_size().at(0);
  src_size.width = image.get_size().at(1);

  NppiRect src_roi = {0, 0, src_size.width, src_size.height};

  NppiSize dst_size;
  dst_size.height = *size.begin();
  dst_size.width = *(size.begin() + 1);

  NppiRect dst_roi = {0, 0, dst_size.width, dst_size.height};

  void* dst_ptr = nullptr;
  if (cudaMalloc(&dst_ptr, dst_size.height * dst_size.width * dst_channels) !=
      cudaSuccess)
    throw std::runtime_error("Cuda malloc error!");

  if (nppiResize_8u_C3R_Ctx(
          reinterpret_cast<Npp8u*>(image.get_data()), src_size.width * 3,
          src_size, src_roi, reinterpret_cast<Npp8u*>(dst_ptr),
          dst_size.width * 3, dst_size, dst_roi, NPPI_INTER_LANCZOS,
          npp_stream_ctx_) != NPP_NO_ERROR)
    throw std::runtime_error("Nppi resize error!");

  return Image({dst_size.height, dst_size.width, 3}, dst_ptr);
}

}  // namespace ipl
