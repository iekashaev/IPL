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

#include "ipl/python/pytorch/PytorchImageResizer.h"

#include "ipl/Image.h"

namespace ipl {
namespace python {
namespace pytorch {

// TODO(Ildar): Test this
void deleter(void* ptr);

PytorchImageResizer::PytorchImageResizer() {}

PytorchImageResizer::~PytorchImageResizer() {}

bool PytorchImageResizer::init() { return resizer_.init(); }

torch::Tensor PytorchImageResizer::resize(const torch::Tensor& image,
                                          const torch::Tensor& size) {
  // Create image object from tensor
  auto im_size = image.sizes();
  Image img({im_size.at(0), im_size.at(1), im_size.at(2)}, image.data_ptr());
  Image resized = resizer_.resize(img, {512, 512});
  auto options =
      c10::TensorOptions(at::kByte).device(torch::Device(at::kCUDA, 0));
  return torch::from_blob(resized.get_data(), {512, 512, img.get_size().at(2)},
                          deleter, options);
}

void init_resizer(const py::module_& m) {
  py::class_<PytorchImageResizer>(m, "ImageResizer")
      .def(py::init<>())
      .def("init", &PytorchImageResizer::init)
      .def("resize", &PytorchImageResizer::resize, py::arg("image"),
           py::arg("size"), py::call_guard<py::gil_scoped_release>());
}

}  // namespace pytorch
}  // namespace python
}  // namespace ipl
