'''
Predict AUD to other currency exchange rate
'''

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
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
normalizer = preprocessing.Normalization(input_shape=[1, ], axis=None)
normalizer.adapt(train_x)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[1, ]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.summary()

# %%
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss='mse')

history = model.fit(
    train_x,
    train_y,
    epochs=200
)

# plot the training history
print(history.history.keys())
loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)


# plot training loss and accuracy
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Predict since day one
prediction = model.predict(train_x)
plt.plot(train_x, prediction, label='Predicted Exchange Rates')
plt.xlabel('Days')
plt.ylabel('Rates')
plt.plot(train_x, train_y, label='Exchange Rates')
plt.xlabel('Days')
plt.ylabel('Rates')
plt.legend()
plt.show()

# Predict rates in a year
predict_days = []
for i in range(365):
    day = total_count + i
    predict_days.append(day)
prediction = model.predict(predict_days)
plt.plot(predict_days, prediction, label='Future Exchange Rates')
plt.xlabel('Days')
plt.ylabel('Rates')
plt.legend()
plt.show()

# %%
