import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the file
filename = "exams/2024-fall-Thor/section2/traffic_train.txt"
df = pd.read_csv(filename, header=None, names=["density", "speed", "weather_type"])

# Split the data into morning and afternoon
morning_df = df.iloc[:140]
afternoon_df = df.iloc[140:]

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(morning_df["density"], morning_df["speed"], color="green", label="Morning Traffic")
plt.scatter(afternoon_df["density"], afternoon_df["speed"], color="blue", label="Afternoon Traffic")

plt.xlabel("Density")
plt.ylabel("Speed")
plt.title("Speed vs Density: Morning and Afternoon Traffic")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
