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

#ifndef IPL_IMAGERESIZER_H_
#define IPL_IMAGERESIZER_H_

#include <npp.h>
#include <initializer_list>
#include "ipl/Image.h"

namespace ipl {

class ImageResizer {
 public:
  ImageResizer();
  ~ImageResizer();

  bool init();
  Image resize(const Image& image, std::initializer_list<int> size);

 private:
  NppStreamContext npp_stream_ctx_;
  cudaStream_t stream_;
};

}  // namespace ipl

#endif  // IPL_IMAGERESIZER_H_
