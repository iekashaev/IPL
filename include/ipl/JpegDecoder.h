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

#ifndef IPL_JPEGDECODER_H_
#define IPL_JPEGDECODER_H_

#include <nvjpeg.h>

#include <string>
#include <vector>

#include "ipl/Image.h"

namespace ipl {

struct ImageInfo {
  int width = 0;
  int height = 0;
  int channels = 0;
  // Encoded data
  std::vector<char> data;
};

class JpegDecoder {
 public:
  JpegDecoder();
  ~JpegDecoder();

  bool init();
  Image read_image(const std::string& image);

 private:
  bool get_image_info(ImageInfo* info);

 private:
  // NVJpeg decode params
  nvjpegHandle_t nvjpeg_handle_;
  nvjpegJpegState_t nvjpeg_state_;
  cudaStream_t stream_;
};

}  // namespace ipl

#endif  // IPL_JPEGDECODER_H_
