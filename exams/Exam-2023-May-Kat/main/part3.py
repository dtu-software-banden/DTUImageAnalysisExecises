
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 

import numpy as np

T1 = np.eye(4)
# ROTATE 30 first
rotation_matrix_1 = rotation_matrix(roll=30, pitch=0, yaw=0, deg = True)
T1 = rotation_matrix_1
T1[0, 3] = 10  

rotation_matrix_2 = rotation_matrix(yaw=10, roll=0, pitch=0, deg = True)
T2 = np.eye(4)
T2 = rotation_matrix_2


T_final = T2 @ T1

print(np.round(T_final, 4))