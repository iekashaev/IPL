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

#ifndef IPL_PYTHON_PYTORCH_PYTORCHJPEGENCODER_H_
#define IPL_PYTHON_PYTORCH_PYTORCHJPEGENCODER_H_

#include <string>

#include "ipl/JpegEncoder.h"
#include "torch/torch.h"

namespace ipl {
namespace python {
namespace pytorch {

class PytorchJpegEncoder {
 public:
  PytorchJpegEncoder();
  ~PytorchJpegEncoder();

  bool init(const int quality);
  void encode_image(const std::string& image_path, const torch::Tensor& image);

 private:
  JpegEncoder _encoder;
};

}  // namespace pytorch
}  // namespace python
}  // namespace ipl

#endif  // IPL_PYTHON_PYTORCH_PYTORCHJPEGENCODER_H_
