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

#include "ipl/JpegDecoder.h"

#include <fstream>

namespace ipl {

static int dev_malloc(void** p, size_t s) {
  return static_cast<int>(cudaMalloc(p, s));
}

static int dev_free(void* p) { return static_cast<int>(cudaFree(p)); }

static int host_malloc(void** p, size_t s, unsigned int f) {
  return static_cast<int>(cudaHostAlloc(p, s, f));
}

static int host_free(void* p) { return static_cast<int>(cudaFreeHost(p)); }

JpegDecoder::JpegDecoder() {}

JpegDecoder::~JpegDecoder() {
  nvjpegJpegStateDestroy(nvjpeg_state_);
  nvjpegDestroy(nvjpeg_handle_);
}

bool JpegDecoder::init() {
  nvjpegDevAllocator_t dev_allocator = {dev_malloc, dev_free};
  nvjpegPinnedAllocator_t pinned_allocator = {host_malloc, host_free};
  if (nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator,
                     NVJPEG_FLAGS_DEFAULT,
                     &nvjpeg_handle_) != NVJPEG_STATUS_SUCCESS)
    return false;

  if (nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_) !=
      NVJPEG_STATUS_SUCCESS)
    return false;

  if (cudaStreamCreateWithFlags(&stream_, cudaStreamDefault) != cudaSuccess)
    return false;

  return true;
}

bool JpegDecoder::get_image_info(ImageInfo* info) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  if (nvjpegGetImageInfo(nvjpeg_handle_,
                         reinterpret_cast<unsigned char*>(info->data.data()),
                         info->data.size(), &channels, &subsampling, widths,
                         heights) != NVJPEG_STATUS_SUCCESS)
    return false;

  info->width = widths[0];
  info->height = heights[0];
  info->channels = channels;

  return true;
}

Image JpegDecoder::read_image(const std::string& image) {
  std::ifstream image_file(image,
                           std::ios::in | std::ios::binary | std::ios::ate);
  if (!image_file.is_open()) throw std::runtime_error("File open error!");

  std::streamsize file_size = image_file.tellg();
  image_file.seekg(0, std::ios::beg);

  ImageInfo image_info;
  // Prepare buffer for image
  image_info.data.resize(file_size);

  if (!image_file.read(image_info.data.data(), file_size))
    throw std::runtime_error("File read error!");

  if (get_image_info(&image_info) == false)
    throw std::runtime_error("Get image info error!");

  void* image_data_dev = nullptr;
  if (dev_malloc(&image_data_dev, image_info.width * image_info.height *
                                      image_info.channels) != cudaSuccess)
    throw std::runtime_error("Cuda malloc error!");

  nvjpegImage_t nvjpeg_image;
  for (int c = 0; c < image_info.channels; c++) {
    int sz = image_info.width * image_info.height;
    nvjpeg_image.channel[c] =
        reinterpret_cast<unsigned char*>(image_data_dev) + (c * sz);
    nvjpeg_image.pitch[c] = static_cast<unsigned int>(image_info.width);
  }
  nvjpeg_image.pitch[0] = static_cast<unsigned int>(image_info.width) * 3;

  if (nvjpegDecode(nvjpeg_handle_, nvjpeg_state_,
                   reinterpret_cast<unsigned char*>(image_info.data.data()),
                   file_size, NVJPEG_OUTPUT_BGRI, &nvjpeg_image,
                   stream_) != NVJPEG_STATUS_SUCCESS)
    throw std::runtime_error("NVJpeg decode error!");

  return Image({image_info.height, image_info.width, image_info.channels},
               image_data_dev, image_info.data);
}

}  // namespace ipl
