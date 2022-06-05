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
cmake ..
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
# License
IPL has a Apache License 2.0 license, as found in the [LICENSE](https://github.com/IldarKashaev/IPL/blob/main/LICENSE) file.
