'''
Predict AUD to other currency exchange rate
'''

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
csv_name = 'AUD2JPY.csv'
csv_path = '../data/{}'.format(csv_name)
if not os.path.exists(csv_path):
    exit('File not found: {}'.format(csv_path))

full_data = pd.read_csv(csv_path)
total_count = full_data.shape[0]
train_x = full_data["Day"]
train_y = full_data["Rate"]
max_rate = max(train_y)
min_rate = min(train_y)
average_rate = np.mean(train_y)
print('Total {} data. Max - {}, Avg - {}, Min - {}'.format(total_count,
      max_rate, average_rate, min_rate))

# Plot the chart
# plt.figure(figsize=(25, 10))
plt.plot(train_x, train_y, label='Exchange Rates')
plt.xlabel('Days')
plt.ylabel('Rates')
plt.legend()
plt.show()

# %%
# Define the model
model = keras.Sequential([
    layers.Dense(units=1)
])
model.summary()

# %%
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='mean_absolute_error')

# %%
history = model.fit(
    all_days,
    all_rates,
    epochs=100
)

# %%
# Predict rates in a year
for i in range(365):
    day = total_count + i
    print('Day {} ({}): {}'.format(day, i + 1, model.predict(day)))
# %%
