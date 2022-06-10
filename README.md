# IPL
IPL is a C++ library for image loading/pre-processing.

# Installation
```shell
git clone -b master --single-branch https://github.com/IldarKashaev/IPL.git
cd IPL
```
#### Pytorch c++ extension for python:
```shell
python3 setup.py install
```

#### Raw c++ library:
```shell
mkdir build
cd build
cmake [-DBUILD_STATIC=ON] ..
make -j$(nproc)
```
# Example
#### Pytorch
```python
import torch
from IPL import JpegDecoder, JpegEncoder

JPEG_IMAGE_PATH = ''
DUMMY_IMAGE_SIZE = (256, 256, 3)
IMAGE_NAME = ''

def decode():
  decoder = JpegDecoder()
  if decoder.init():
    image = decoder.read_image(JPEG_IMAGE_PATH)
    ...
  else:
    print('Can`t init decoder')

def encode():
  encoder = JpegEncoder()
  quality = 100
  cuda_device = torch.device('cuda:0')
  if encoder.init(quality):
    dummy_image = torch.randint(0, 255, DUMMY_IMAGE_SIZE,
                                dtype=torch.uint8, device=cuda_device)
    encoder.encode_image(IMAGE_NAME, dummy_image)
    ...
  else:
    print('Can`t init encoder')

def main():
  print('Decode')
  decode()
  print('Encode')
  encode()

if __name__ == '__main__':
  main()
```
#### C++
```cc
#include <iostream>
#include "ipl/JpegDecoder.h"
#include "ipl/JpegEncoder.h"

#define JPEG_IMAGE_PATH ""
#define DUMMY_IMAGE_SIZE {256, 256, 3}
#define IMAGE_NAME ""

void decode() {
  ipl::JpegDecoder decoder;
  if (decoder.init()) {
    auto image = decoder.read_image(JPEG_IMAGE_PATH);
  } else {
    std::cout << "Can`t init decoder" << std::endl;
  }
}

void encode() {
  ipl::JpegEncoder encoder;
  const int quality = 100;
  if (encoder.init(quality)) {
    ipl::Image dummy_image(DUMMY_IMAGE_SIZE, device_ptr);
    encoder.encode_image(IMAGE_NAME, dummy_image);
  } else {
    std::cout << "Can`t init encoder" << std::endl;
  }
}

int main() {
  std::cout << "Decode" << std::endl;
  decode();
  std::cout << "Encode" << std::endl;
  encode();

  return 0;
}
```
# License
IPL has a Apache License 2.0 license, as found in the [LICENSE](https://github.com/IldarKashaev/IPL/blob/main/LICENSE) file.
