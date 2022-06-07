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
#include "torch/torch.h"

namespace ipl {
namespace python {
namespace pytorch {

class PytorchImageResizer {
 public:
  PytorchImageResizer();
  ~PytorchImageResizer();

  bool init();
  torch::Tensor resize(const torch::Tensor& image, const torch::Tensor& size);

 private:
  ImageResizer resizer_;
};

}  // namespace pytorch
}  // namespace python
}  // namespace ipl
