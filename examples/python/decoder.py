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
from IPL import JpegDecoder

IMAGE_PATH = '../images/cat.jpg'

def main():
  ''' Decoder example '''
  # Create decoder object
  decoder = JpegDecoder()
  # Init decoder
  if decoder.init():
    # Read image into GPU
    image = decoder.read_image(IMAGE_PATH) # pylint: disable=unused-variable
    ...
  else:
    print('Can`t init decoder')

if __name__ == '__main__':
  main()
