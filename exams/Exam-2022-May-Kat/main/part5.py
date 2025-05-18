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


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

with open("./data/soccer_data.txt", "r") as f:
    raw_lines = f.readlines()

clean_lines = [line.strip() for line in raw_lines if not line.startswith("%")]
data = [list(map(float, line.split())) for line in clean_lines]
df = pd.DataFrame(data, columns=[
    "short_passing", "long_passing", "acceleration",
    "sprint_speed", "agility", "stamina"
])

pca_raw = PCA()
projected_raw = pca_raw.fit_transform(df)

def Question1():
    print("Running Question 1")
    max_abs_value = np.max(np.abs(projected_raw))
    print(max_abs_value)


def Question2():
    print("Running Question 2")
    print(pca_raw.explained_variance_ratio_)


def Question3():
    print("Running Question 3")


def Question4():
    print("Running Question 4")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()