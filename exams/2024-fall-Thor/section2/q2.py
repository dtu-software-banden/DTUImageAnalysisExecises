import numpy as np
import pandas as pd

from utils.lda_utils import lda_predict_batch, lda_train

# Load the data from the file
filename = "exams/2024-fall-Thor/section2/traffic_train.txt"
df = pd.read_csv(filename, header=None, names=["density", "speed", "weather_type"])

# Split the data into morning and afternoon
morning_df = df.iloc[:140].drop(columns=["weather_type"])
afternoon_df = df.iloc[140:].drop(columns=["weather_type"])

morning_model,afternoon_model,sigma = lda_train(morning_df,afternoon_df)

filename = "exams/2024-fall-Thor/section2/traffic_test.txt"
test_df = pd.read_csv(filename, header=None, names=["density", "speed", "weather_type"])
test_raw = test_df.drop(columns=["weather_type"])


data,prediction = lda_predict_batch(test_raw,morning_model,afternoon_model,sigma)
afternoon_pred = prediction[60:]
print(np.where((afternoon_pred < 2),1,0).sum())
