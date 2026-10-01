import numpy as np
import pandas as pd
import yaml
import tensorflow as tf


# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

dense_units = params["train"]["dense_units"]
dropout_rate = params["train"]["dropout_rate"]
learning_rate = params["train"]["learning_rate"]
epochs = params["train"]["epochs"]
batch_size = params["train"]["batch_size"]


# Load processed data
x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")


# Build ANN
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(dense_units, activation="relu"),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(10, activation="softmax")
])


# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)


# Save model
model.save("models/model.h5")


# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv("models/history.csv", index=False)

print("Training completed successfully.")