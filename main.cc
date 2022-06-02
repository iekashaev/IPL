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

#include <iostream>
#include "ipl/JpegDecoder.h"

int main() {
  JpegDecoder decoder;
  const std::string image = "";
  if (decoder.init() == true) {
    void* img = decoder.read_image(image);
    cudaFree(img);
  } else {
    std::cout << "Decoder init error!\n";
  }
  return 0;
}
