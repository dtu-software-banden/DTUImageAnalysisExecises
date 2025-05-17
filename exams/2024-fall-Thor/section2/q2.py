import pandas as pd

# Load the data from the file
filename = "exams/2024-fall-Thor/section2/traffic_train.txt"
df = pd.read_csv(filename, header=None, names=["density", "speed", "weather_type"])

# Split the data into morning and afternoon
morning_df = df.iloc[:140]
afternoon_df = df.iloc[140:]

