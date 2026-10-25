import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report

from get_dataset import get_ds

train, x_train, y_train, valid, x_valid, y_valid, test, x_test, y_test = get_ds()


def plot_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    fig.savefig(os.path.join('plots', filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_model(x_train, y_train, nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                     metrics=['accuracy'])

    history = nn_model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.2
    )

    return nn_model, history


min_loss = float('inf')
best_model = None
epochs = 100
for n in [16, 32, 64]:
    for probs in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for size in [32, 64, 128]:
                print(f"{n=}, {probs=}, {lr=}, {size=}")
                filename = f"{n=}_{probs=}_{lr=}_{size=}.png"

                model, history = train_model(x_train, y_train, n, probs, lr, size, epochs)
                plot_history(history, filename)

                val_loss = model.evaluate(x_valid, y_valid, verbose=0)[0]
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_model = model

y_pred = best_model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1, )
print(classification_report(y_test, y_pred))
