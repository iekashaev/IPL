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

#include "ipl/python/pytorch/PytorchJpegDecoder.h"

namespace ipl {
namespace python {
namespace pytorch {

// TODO(Ildar): Test this
void deleter(void* ptr) { cudaFree(ptr); }

PytorchJpegDecoder::PytorchJpegDecoder() {}

PytorchJpegDecoder::~PytorchJpegDecoder() {}

bool PytorchJpegDecoder::init() { return _decoder.init(); }

torch::Tensor PytorchJpegDecoder::read_image(const std::string& image) {
  auto im = _decoder.read_image(image);
  auto options =
      c10::TensorOptions(at::kByte).device(torch::Device(at::kCUDA, 0));
  return torch::from_blob(
      im.get_data(),
      {im.get_size().at(0), im.get_size().at(1), im.get_size().at(2)}, deleter,
      options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PytorchJpegDecoder>(m, "JpegDecoder")
      .def(py::init<>())
      .def("init", &PytorchJpegDecoder::init)
      .def("read_image", &PytorchJpegDecoder::read_image, py::arg("image"),
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace pytorch
}  // namespace python
}  // namespace ipl
