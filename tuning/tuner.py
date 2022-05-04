##############################################################################
# The Keras Tuner is a library that helps you pick the optimal set of
# hyperparameters for your TensorFlow program
#
##############################################################################

import tensorflow as tf
from tensorflow import keras

import keras_tuner as kt

(img_train, label_train), (
    img_test,
    label_test,
) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
img_train = img_train.astype("float32") / 255.0
img_test = img_test.astype("float32") / 255.0

## Define the model
def model_build(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


## Instantiate the tuner and perform hypertuning
# The Hyperband tuning algorithm uses resource allocation and early-stopping
# to quickly converge on a high-performing model. This is done using a sports
# championship style bracket. The algorithm trains a large number of models
# for a few epochs and carries forward only the top-performing half of models
# to the next round.
tuner = kt.Hyperband(
    model_build,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="my_dir",
    project_name="intro_to_kt",
)

# Create a callback to stop training early after reaching a certain
# value for the validation loss
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Run the hyperparameter search
tuner.search(
    img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early]
)
# Get the optimal hyperparameters
best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
print(
    f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hyperparams.get('units')} and the optimal learning rate for the optimizer
is {best_hyperparams.get('learning_rate')}.
"""
)

## Train the model
model = tuner.hypermodel.build(best_hyperparams)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history["val_accuracy"]
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print("Best epoch: %d" % (best_epoch,))

## Re-instantiate the hypermodel and train it with the optimal number of
## epochs from above
hypermodel = tuner.hypermodel.build(best_hyperparams)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

### Evaluate the hypermodel on the test data
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)
