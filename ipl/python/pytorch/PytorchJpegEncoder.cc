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

#include "ipl/python/pytorch/PytorchJpegEncoder.h"

#include "ipl/Image.h"

namespace ipl {
namespace python {
namespace pytorch {

PytorchJpegEncoder::PytorchJpegEncoder() {}

PytorchJpegEncoder::~PytorchJpegEncoder() {}

bool PytorchJpegEncoder::init(const int quality) {
  return _encoder.init(quality);
}

void PytorchJpegEncoder::encode_image(const std::string& image_path,
                                      const torch::Tensor& image) {
  if (!image.is_cuda())
    throw std::runtime_error("The image must be on the cuda device!");

  // Create image object from tensor
  auto size = image.sizes();
  Image img({size.at(0), size.at(1), size.at(2)}, image.data_ptr());
  _encoder.encode_image(image_path, img);
}

void init_encoder(const py::module_& m) {
  py::class_<PytorchJpegEncoder>(m, "JpegEncoder")
      .def(py::init<>())
      .def("init", &PytorchJpegEncoder::init, py::arg("quality"))
      .def("encode_image", &PytorchJpegEncoder::encode_image,
           py::arg("image_path"), py::arg("image"),
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace pytorch
}  // namespace python
}  // namespace ipl
