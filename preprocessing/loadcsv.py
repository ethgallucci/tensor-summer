### https://www.tensorflow.org/tutorials/load_data/csv

import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=[
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Age",
    ],
)

print(abalone_train.head())

# Seperate features and labels for training
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop("Age")

# Pack the features into a single numpy array
abalone_features = np.array(abalone_features)
print(abalone_features)

### Basic preprocessing

# It's good practice to normalize the inputs to your model. The Keras
# preprocessing layers provide a convenient way to build this normalization
# into your model
normalize = keras.layers.Normalization()
normalize.adapt(abalone_features)

normalized_abalone_model = tf.keras.Sequential(
    [normalize, keras.layers.Dense(64), keras.layers.Dense(1)]
)

normalized_abalone_model.compile(
    loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam()
)

normalized_abalone_model.fit(abalone_features, abalone_labels, epochs=10)
