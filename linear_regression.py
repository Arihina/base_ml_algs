import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
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


_, x_train_temp, y_train_temp = get_xy(df, "Rented Bike Count", x=["Temperature"])
_, x_val_temp, y_val_temp = get_xy(df, "Rented Bike Count", x=["Temperature"])
_, x_test_temp, y_test_temp = get_xy(df, "Rented Bike Count", x=["Temperature"])

temp_reg = LinearRegression()
temp_reg.fit(x_train_temp, y_train_temp)

print(temp_reg.coef_, temp_reg.intercept_)
print(temp_reg.score(x_test_temp, y_test_temp))

plt.scatter(x_train_temp, y_train_temp, label="Data", color="Blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bike vs Temp")
plt.ylabel("Num of bike")
plt.xlabel("Temp")
plt.show()

# Multiple LR

_, x_train_all, y_train_all = get_xy(df, "Rented Bike Count", x=df.columns[1:])
_, x_val_all, y_val_all = get_xy(df, "Rented Bike Count", x=df.columns[1:])
_, x_test_all, y_test_all = get_xy(df, "Rented Bike Count", x=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(x_train_all, y_train_all)

print(all_reg.coef_, all_reg.intercept_)
print(all_reg.score(x_test_all, y_test_all))

