# Model progress can be saved during and after training. This means a model can resume where it left off and avoid long
# training times. Saving also means you can share your model and others can recreate your work.

import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.Sequential(
        [
            keras.layers.Dense(512, activation="relu", input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    return model


model = create_model()
model.summary()

checkpoint_path = "./checkpoint/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

# Train the model with the new callback
# This may generate warnings related to saving the state of the optimizer.
# These warnings are in place to discourage outdated usage, and can be ignored.
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback],
)
