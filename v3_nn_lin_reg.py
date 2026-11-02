import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from ucimlrepo import fetch_ucirepo

seoul_bike_sharing_demand = fetch_ucirepo(id=560)

x = seoul_bike_sharing_demand.data.features
y = seoul_bike_sharing_demand.data.targets

df = pd.concat([x, y], axis=1).drop(["Date", "Holiday", "Seasons"], axis=1)
df['Functioning Day'] = (df['Functioning Day'] == 'Yes').astype(int)
df = df[df['Hour'] == 12]
df = df.drop(['Hour'], axis=1)
df = df.drop(["Wind speed", "Visibility", "Functioning Day"], axis=1)

# print(df.columns)

train, val, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


def get_xy(df, y, x=None):
    df = copy.deepcopy(df)
    if x is None:
        x = df[[c for c in df.columns if c != y]].values
    else:
        if len(x) == 1:
            x = df[x[0]].values.reshape(-1, 1)
        else:
            x = df[x].values

    y = df[y].values.reshape(-1, 1)
    data = np.hstack((x, y))

    return data, x, y


_, x_train_all, y_train_all = get_xy(df, "Rented Bike Count", x=df.columns[1:])
_, x_val_all, y_val_all = get_xy(df, "Rented Bike Count", x=df.columns[1:])
_, x_test_all, y_test_all = get_xy(df, "Rented Bike Count", x=df.columns[1:])

all_normalizer = tf.keras.layers.Normalization(input_shape=(6, 1), axis=-1)
all_normalizer.adapt(x_train_all.reshape(-1))

model = tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

history = model.fit(
    x_train_all, y_train_all,
    epochs=1000,
    validation_data=(x_val_all, y_val_all)
)

plt.scatter(x_train_all, y_train_all, label="Data", color="Blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bike vs Temp")
plt.ylabel("Num of bike")
plt.xlabel("Temp")
plt.show()

plt.figure(figsize=(10, 6))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)

plt.show()
