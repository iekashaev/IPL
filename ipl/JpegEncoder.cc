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

#include "ipl/JpegEncoder.h"

#include <fstream>
#include <iostream>

namespace ipl {

JpegEncoder::JpegEncoder() {}

JpegEncoder::~JpegEncoder() {}

bool JpegEncoder::init(const int quality) {
  if (nvjpegCreateSimple(&nvjpeg_handle_) != NVJPEG_STATUS_SUCCESS)
    return false;

  if (cudaStreamCreateWithFlags(&stream_, cudaStreamDefault) != cudaSuccess)
    return false;

  if (nvjpegEncoderStateCreate(nvjpeg_handle_, &nvjpeg_state_, stream_) !=
      NVJPEG_STATUS_SUCCESS)
    return false;

  if (nvjpegEncoderParamsCreate(nvjpeg_handle_, &nvjpeg_params_, stream_) !=
      NVJPEG_STATUS_SUCCESS)
    return false;

  if (nvjpegEncoderParamsSetSamplingFactors(nvjpeg_params_, NVJPEG_CSS_444,
                                            0) != NVJPEG_STATUS_SUCCESS)
    return false;

  if (nvjpegEncoderParamsSetQuality(nvjpeg_params_, quality, stream_) !=
      NVJPEG_STATUS_SUCCESS)
    return false;

  return true;
}

void JpegEncoder::write_to_file(const std::string& image_path,
                                void* encoded_data, size_t len) {
  std::ofstream output_file(image_path, std::ios::out | std::ios::binary);
  if (!output_file.is_open()) throw std::runtime_error("Open file error!");
  output_file.write(reinterpret_cast<char*>(encoded_data), len);
  output_file.close();
}

void JpegEncoder::encode_image(const std::string& image_path,
                               const Image& image) {
  int height = image.get_size().at(0);
  int width = image.get_size().at(1);
  int channels = image.get_size().at(2);

  nvjpegImage_t nvjpeg_image;
  nvjpeg_image.channel[0] = reinterpret_cast<unsigned char*>(image.get_data());
  nvjpeg_image.pitch[0] = static_cast<unsigned int>(width) * 3;

  if (nvjpegEncodeImage(nvjpeg_handle_, nvjpeg_state_, nvjpeg_params_,
                        &nvjpeg_image, NVJPEG_INPUT_RGBI, width, height,
                        stream_) != NVJPEG_STATUS_SUCCESS)
    throw std::runtime_error("NVJpeg encode error!");

  size_t length = 0;

  if (nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, nvjpeg_state_, NULL,
                                    &length, stream_) != NVJPEG_STATUS_SUCCESS)
    throw std::runtime_error("NVJpeg retrieve bitstream error!");

  cudaStreamSynchronize(stream_);

  std::vector<unsigned char> jpeg(length);
  if (nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, nvjpeg_state_, jpeg.data(),
                                    &length, stream_) != NVJPEG_STATUS_SUCCESS)
    throw std::runtime_error("NVJpeg retrieve bitstream error!");
  cudaStreamSynchronize(stream_);
  write_to_file(image_path, jpeg.data(), length);
}

}  // namespace ipl
