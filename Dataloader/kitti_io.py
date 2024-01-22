from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys


# Read Image
def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img