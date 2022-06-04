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

#ifndef IPL_JPEGENCODER_H_
#define IPL_JPEGENCODER_H_

#include <nvjpeg.h>

#include <string>

#include "ipl/Image.h"

namespace ipl {

class JpegEncoder {
 public:
  JpegEncoder();
  ~JpegEncoder();

  bool init(const int quality);
  void encode_image(const std::string& image_path, const Image& image);

 private:
  void write_to_file(const std::string& image_path, void* encoded_data,
                     size_t len);

 private:
  // NVJpeg encode params
  nvjpegHandle_t nvjpeg_handle_;
  nvjpegEncoderState_t nvjpeg_state_;
  nvjpegEncoderParams_t nvjpeg_params_;
  cudaStream_t stream_;
};

}  // namespace ipl

#endif  // IPL_JPEGENCODER_H_
