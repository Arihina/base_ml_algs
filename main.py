import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

magic_gamma_telescope = fetch_ucirepo(id=159)

x = magic_gamma_telescope.data.features
y = magic_gamma_telescope.data.targets

df = pd.concat([x, y], axis=1)
df['class'] = (df['class'] == 'g').astype(int)


def get_plt():
    for label in df.columns[:-1]:
        plt.hist(df[df['class'] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
        plt.hist(df[df['class'] == 0][label], color='red', label='hadron', alpha=0.7, density=True)

        plt.ylabel('Probability')
        plt.title(label)
        plt.xlabel(label)
        plt.legend()
        plt.show()


df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train = df.iloc[:int(0.6 * len(df))]
valid = df.iloc[int(0.6 * len(df)):int(0.8 * len(df))]
test = df.iloc[int(0.8 * len(df)):]


def scale_dataset(dataframe, oversampling=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversampling:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y


train, x_train, y_train = scale_dataset(train, oversampling=True)
valid, x_valid, y_valid = scale_dataset(valid, oversampling=False)
test, x_test, y_test = scale_dataset(test, oversampling=False)
