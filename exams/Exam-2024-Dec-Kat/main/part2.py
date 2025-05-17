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
from utils.lda_utils import *


import numpy as np
import pandas as pd
import os

# === Define paths ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_path = os.path.join(base_path, "data", "traffic", "traffic_train.txt")
test_path = os.path.join(base_path, "data", "traffic", "traffic_test.txt")

# === Load data ===
train_data = pd.read_csv(train_path, header=None, names=["density", "speed", "weather"], sep=",")
test_data = pd.read_csv(test_path, header=None, names=["density", "speed", "weather"], sep=",")

# === Add labels ===
train_data["label"] = ["morning"] * 140 + ["afternoon"] * 140
test_data["label"] = ["morning"] * 60 + ["afternoon"] * 60

# === Print sample ===
print(train_data.head())



def plot_traffic_data(df):
    morning = df[df["label"] == "morning"]
    afternoon = df[df["label"] == "afternoon"]

    plt.figure(figsize=(8, 6))
    plt.scatter(morning["density"], morning["speed"], color="green", label="Morning", alpha=0.6)
    plt.scatter(afternoon["density"], afternoon["speed"], color="blue", label="Afternoon", alpha=0.6)
    plt.xlabel("Density (number of cars)")
    plt.ylabel("Speed (Km/h)")
    plt.title("Traffic Speed vs. Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def Question1():
    print("Running Question 1")
    plot_traffic_data(train_data)



def Question2():
    print("Running Question 2")

    # Prepare train/test features
    X_train = train_data[["density", "speed"]].values
    y_train = np.array([1] * 140 + [2] * 140)  # 1 = morning, 2 = afternoon
    X_test = test_data[["density", "speed"]].values
    y_test = np.array([1] * 60 + [2] * 60)

    # Train LDA
    X1 = X_train[y_train == 1]
    X2 = X_train[y_train == 2]
    mu1, mu2, sigma = lda_train(X1, X2)

    # Predict on test set
    scores, y_pred = lda_predict_batch(X_test, mu1, mu2, sigma)

    # Count afternoon misclassified as morning
    afternoon = y_test == 2
    misclassified = np.sum((y_pred == 1) & afternoon)
    print(f"Afternoon samples misclassified as morning: {misclassified}")


def Question3():
    print("Running Question 3")

    # First 140 rows = morning
    morning_data = train_data.iloc[:140]

    # Count rainy days (weather == 1)
    rainy_mornings = (morning_data["weather"] == 1).sum()
    print(f"Number of rainy mornings: {rainy_mornings}")

def Question4():
    print("Running Question 4")
    print("Option A")


if __name__ == "__main__":
    Question1()
    Question2()
    Question3()
    Question4()