### https://www.tensorflow.org/tutorials/generative/deepdream

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

import numpy as np
import matplotlib as plt
import PIL.Image

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"


def download(url, max_dim=None):
    name = url.split("/")[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


def show(img):
    PIL.Image.fromarray(np.array(img)).show()


original_img = download(url, max_dim=500)
show(original_img)

## Download and prepare a pre-trained image classification model
base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")

# The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in
# a way that the image increasingly "excites" the layers. The complexity of the features
# incorporated depends on layers chosen by you, i.e, lower layers produce strokes or simple
# patterns, while deeper layers give sophisticated features in images, or even whole objects.

## Maximize the activations of the layers
names = ["mixed3", "mixed5"]
layers = [base_model.get_layer(name).output for name in names]

## Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

### Calculate loss
def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


### Gradient Ascent

# Once you have calculated the loss for the chosen layers, all that is left is to calculate
# the gradients with respect to the image, and add them to the original image. Adding the gradients
# to the image enhacnes the patterns seen by the network. At each step, you will have created
# an image that increasingly excites the activations of certain layers in the network


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)

            gradients = tape.gradient(loss, img)

            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


deepdream = DeepDream(dream_model)


### Main loop
def run_deep_dream(img, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    show(result)

    return result


dream_img = run_deep_dream(img=original_img, steps=100, step_size=0.06)
show(dream_img)
