"""
Predict AUD to other currency exchange rate
"""

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os

# %%
# Load data from csv
csv_name = "AUD2JPY.csv"
csv_path = "../data/{}".format(csv_name)
if not os.path.exists(csv_path):
    exit("File not found: {}".format(csv_path))

full_data = pd.read_csv(csv_path)
total_count = full_data.shape[0]
all_days = full_data.iloc[:, 0].values
all_rates = full_data.iloc[:, 1].values
print("Total {} data".format(total_count))

# Plot the chart
plt.plot(all_days, all_rates, label='Exchange Rates')
plt.xlabel('Days')
plt.ylabel('Rates')
plt.legend()
plt.show()

# %%
