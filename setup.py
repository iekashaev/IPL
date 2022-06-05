# Copyright (c) 2022, Ildar Kashaev. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension
import torch


def build(): # pylint: disable=missing-function-docstring
  library_path = torch.utils.cpp_extension.library_paths(cuda=True)
  include_path = torch.utils.cpp_extension.include_paths(cuda=True)

  app_src_path = ['ipl/Image.cc',
                  'ipl/JpegDecoder.cc',
                  'ipl/JpegEncoder.cc',
                  'ipl/python/pytorch/Ext.cc',
                  'ipl/python/pytorch/PytorchJpegDecoder.cc',
                  'ipl/python/pytorch/PytorchJpegEncoder.cc',]

  library = ['cuda']
  library += ['nvjpeg']
  library += ['torch']
  library += ['torch_cpu']
  library += ['torch_cuda']
  library += ['torch_python']
  library += ['c10']
  library += ['c10_cuda']

  include_path += [os.getcwd() + '/include']

  setup(
    name='IPL',
    version=0.0,
    author='Ildar Kashaev',
    ext_modules=[
      Extension(
        name='IPL',
        sources=app_src_path,
        include_dirs=include_path,
        library_dirs=library_path,
        libraries=library,
        extra_compile_args=['-g', '-std=c++17'],
        language='c++')
    ],
    cmdclass={
      'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(),
    zip_safe=True,
    python_requires='>=3.6',
    install_requires=['torch'],
  )

if __name__ == '__main__':
  build()
