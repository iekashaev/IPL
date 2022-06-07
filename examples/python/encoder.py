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

import torch
from IPL import JpegEncoder

DUMMY_IMAGE_SIZE = (256, 256, 3)
IMAGE_NAME = './dummy.jpg'

def main():
  ''' Encoder example '''
  # Create encoder object
  encoder = JpegEncoder()
  quality = 100
  cuda_device = torch.device('cuda:0')
  # Init encoder
  if encoder.init(quality):
    dummy_image = torch.randint(0, 255, DUMMY_IMAGE_SIZE,
                                dtype=torch.uint8, device=cuda_device)
    encoder.encode_image(IMAGE_NAME, dummy_image)
    ...
  else:
    print('Can`t init encoder')

if __name__ == '__main__':
  main()
