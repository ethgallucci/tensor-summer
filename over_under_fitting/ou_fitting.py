#################
# In the './classify_clothing.py' and './regression/*.py' models, the accuracy of the models
# on the validation data would peak after training for a number of epochs and then stagnate or start
# decreasing.
#
# In other words, the model would 'overfit' to the data. Learning how to deal with overfitting is important. Although
# it's often possible to achieve high accuracy on the training set, what you really want is to develop models
# that generalize well to a testing set (or data they haven't seen before)
#
# The opposite of overfitting is underfitting. Underfitting occurs when there is still room for improvement on the training data. This
# can happen for a number of reasons: If the model is not powerful enough, is over-regularized, or has simply not been trained long enough.
# This means the network has not learned the relevant patterns in the training data.
#
# If you train for too long though, the model will start to overfit and learn patterns from the training data that don't generalize to the
# test data. You need to strike a balance. Understanding how to train for an appropriate number of epochs as you'll explore below is a useful skill.
#
# To prevent overfitting, the best solution is to use more complete training data. The dataset should cover the full range of inputs that the model is
# expected to handle. Additional data may only be useful if it covers new and interesting cases.
#
# A model trained on more complete data will naturally generalize better. When that is no longer possible, the next best solution is to use techniques
# like regularization. These place constraints on the quantity and type of information your model can store. If a network can only afford to memorize a
# small number of patterns, the optimization process will force it to focus on the most prominent patterns, which have a better chance of generalizing well.
################

import tensorflow as tf
from tensorflow.keras import layers, regularizers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile


logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

### The Higgs Dataset
gz = tf.keras.utils.get_file(
    "HIGGS.csv.gz", "http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz"
)

FEATURES = 28
ds = tf.data.experimental.CsvDataset(
    gz,
    [
        float(),
    ]
    * (FEATURES + 1),
    compression_type="GZIP",
)


def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


packed_ds = ds.batch(10000).map(pack_row).unbatch()

# Use just the first 1000 samples for validation, and the next 10000 for training
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
print(train_ds)

# These datasets return individual examples. Use the batch method to create batches
# of an appropriate size for training.
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

### Demonstrate Overfitting #######################

# Training procedure
## Hyperbolically decrease the learning rate to 1/2 the base rate at 1000 epochs, 1/3 at 2000 epochs, and so on
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1, staircase=False
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step / STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel("Epoch")
_ = plt.ylabel("Learning Rate")
plt.savefig("./over_under_fitting/learning_rate_epoch.png")


def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_crossentropy", patience=200
        ),
        tf.keras.callbacks.TensorBoard(logdir / name),
    ]


def compile_and_fit_model(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.losses.BinaryCrossentropy(
                from_logits=True, name="binary_crossentropy"
            ),
            "accuracy",
        ],
    )

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=1,
    )

    return history


## Tiny model
tiny_model = tf.keras.Sequential(
    [layers.Dense(16, activation="elu", input_shape=(FEATURES,)), layers.Dense(1)]
)

size_histories = {}
size_histories["Tiny"] = compile_and_fit_model(tiny_model, "sizes/Tiny")

plotter = tfdocs.plots.HistoryPlotter(metric="binary_crossentropy", smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.savefig("./over_under_fitting/tiny_model.png")
plt.clf()
