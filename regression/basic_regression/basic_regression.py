import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# Get the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

raw_dataset = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

# Clean the data
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

# One-hot encode the values in the column with pd.get_dummies
dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

# Split the data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
sns_plot = sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
).savefig("./regression/clean_data_sns.png")

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")

# Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print("\n\nFirst example: ", first)
    print()
    print("Normalized: ", normalizer(first).numpy(), "\n")


### Single-variable Linear Regression

# Normalize the Horsepower input features
horsepower = np.array(train_features["Horsepower"])
horsepower_normalizer = layers.Normalization(
    input_shape=[
        1,
    ],
    axis=None,
)
horsepower_normalizer.adapt(horsepower)

# Define the model architecture
horsepower_model = tf.keras.Sequential([horsepower_normalizer, layers.Dense(units=1)])
horsepower_model.summary()

# Compile the model
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error"
)

# Train for 100 epochs
history = horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    epochs=100,
    # Suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split=0.2,
)

# Visualize the model's training progress
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print("\n\nAfter 100 epochs: ", hist.tail())


def plot_loss(history, filename):
    path2file = f"./regression/basic_regression/{filename}"
    plt.clf()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)
    plt.savefig(path2file)


plot_loss(history, "single_var_loss.png")

# Collect the results on the test set for later
test_results = {}
test_results["horsepower_model"] = horsepower_model.evaluate(
    test_features["Horsepower"], test_labels, verbose=0
)

# Since this is single variable regression, it's easy to view the model's predictions as a function of the input
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y):
    plt.clf()
    plt.scatter(train_features["Horsepower"], train_labels, label="Data")
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.legend()
    plt.savefig("./regression/basic_regression/predictions.png")


plot_horsepower(x, y)
