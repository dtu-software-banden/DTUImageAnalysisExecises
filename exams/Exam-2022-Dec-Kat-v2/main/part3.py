import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 
from utils.analysis_utils import * 

import numpy as np

# Original image matrix
img = np.array([
    [64, 94, 21, 19, 31],
    [38, 88, 30, 23, 92],
    [81, 55, 47, 17, 43],
    [53, 62, 23, 23, 18],
    [35, 59, 84, 44, 90]
])

acc = accumulator_image(img)

# Minimal value in the last row
min_val_last_row = np.min(acc[-1])
print("Minimal value in the last row:", min_val_last_row)

black = 178 + 60 + 155 +252
white = (168 + 217 + 159 + 223) + (97 + 136 + 32 + 108)
print("Haar feature:", black - white)


data = np.array([
    [ 33,  12, 110, 122, 204, 218,  25, 231],
    [200,  53,  81, 187, 145, 135, 221, 169],
    [220, 120, 107,   6,  39,  12, 108, 201],
    [114, 168, 217, 178,  60,  97, 136,  16],
    [253, 159, 223, 155, 252,  32, 108,  86],
    [131,  68,  68,  69, 129, 244, 174, 119],
    [ 93,  51,  45, 122,  44, 105,  71,  15],
    [ 75, 149,  45, 233,  24,  64, 146,  56]
])

print(integral_image(data)[2][2])