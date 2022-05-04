### https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

ds_url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
csv_file = "datasets/petfinder-mini/petfinder-mini.csv"

tf.keras.utils.get_file("petfinder_mini.zip", ds_url, extract=True, cache_dir=".")
dataframe = pd.read_csv(csv_file)
print(dataframe.head())

dataframe["target"] = np.where(dataframe["AdoptionSpeed"] == 4, 0, 1)
dataframe = dataframe.drop(columns=["AdoptionSpeed", "Description"])

train, val, test = np.split(
    dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))]
)
print(len(train), "training examples")
print(len(val), "validation examples")
print(len(test), "test examples")

## Create an input pipeline using tf.data
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop("target")
    df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)

    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print("Every feature:", list(train_features.keys()))
print("A batch of ages:", train_features["Age"])
print("A batch of targets:", label_batch)

######################################
# Apply the Keras preprocessing layers
######################################


def get_normalization_layer(name, ds):
    normalizer = keras.layers.Normalization(axis=None)
    feature_ds = ds.map(lambda x, _: x[name])
    normalizer.adapt(feature_ds)
    return normalizer


photo_count_col = train_features["PhotoAmt"]
layer = get_normalization_layer("PhotoAmt", train_ds)
print(layer(photo_count_col))

## Categorical columns
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == "string":
        index = keras.layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = keras.layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


test_type_col = train_features["Type"]
test_type_layer = get_category_encoding_layer(
    name="Type", dataset=train_ds, dtype="string"
)
print(test_type_layer(test_type_col))

test_age_col = train_features["Age"]
test_age_layer = get_category_encoding_layer(
    name="Age", dataset=train_ds, dtype="int64", max_tokens=5
)
print(test_age_layer(test_age_col))


################################################################
# Preprocess selected features to train the model on
################################################################

batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
validation_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Normalize the numerical features
all_inputs = []
encoded_features = []

# Numerical features.
for header in ["PhotoAmt", "Fee"]:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

### Turn the integer categorical values from the dataset (the pet age) into
### integer indices, perform multi-hot encoding, and add the resulting feature
### inputs to encoded_features
age_col = tf.keras.Input(shape=(1,), name="Age", dtype="int64")

encoding_layer = get_category_encoding_layer(
    name="Age", dataset=train_ds, dtype="int64", max_tokens=5
)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)

### Do the same for the string categorical values
categorical_cols = [
    "Type",
    "Color1",
    "Color2",
    "Gender",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Sterilized",
    "Health",
    "Breed1",
]

for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype="string")
    encoding_layer = get_category_encoding_layer(
        name=header, dataset=train_ds, dtype="string", max_tokens=5
    )
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)


#######################################
# Create, compile, and train the model
#######################################

all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Use `rankdir='LR'` to make the graph horizontal.
tf.keras.utils.plot_model(
    model, show_shapes=True, rankdir="LR", to_file="fmp_model.png"
)

model.fit(train_ds, epochs=10, validation_data=validation_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

### Perform inference

# The model you have developed can now classify a row from a CSV file directly
# after you've included the preprocessing layers inside the model itself.
#
# You can now save and reload the Keras model with Model.save and Model.load_model
# before performing inference on new data

model.save("my_pet_classifier")
reloaded_model = tf.keras.models.load_model("my_pet_classifier")
