import os

# Load compressed models from tensorflow_hub
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False

import numpy as np
import PIL.Image
import time
import functools
from pprint import pprint

import tensorflow as tf

print(tf.__version__)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


content_path = tf.keras.utils.get_file(
    "YellowLabradorLooking_new.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
)
style_path = tf.keras.utils.get_file(
    "kandinsky5.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
)

# Visualizing the input
## Define a function to load an image and limit its maximum dimension to 512 pixels
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# Create a simple function to dislpay an image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, "Content Image")
plt.subplot(1, 2, 2)
imshow(style_image, "Style Image")


# Define content and style representations
x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights="imagenet")
prediction_probabilities = vgg(x)
print(prediction_probabilities.shape)

predicted_top5 = tf.keras.applications.vgg19.decode_predictions(
    prediction_probabilities.numpy()
)[0]
pprint([(class_name, prob) for (number, class_name, prob) in predicted_top5])

## Now load a VGG19 without the classification head, and list the layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
print()
for layer in vgg.layers:
    print(layer.name)

## Choose intermediate layers from the network to represent the style and content of the image
content_layers = ["block5_conv2"]
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

"""Intermediate layers for style and content:

So why do these intermediate outputs within our pretrained image classification network allow us to define style and content representations?
At a high level, in order for a network to perform image classification (which this network has been trained to do), it must understand the image. 
This requires taking the raw image as input pixels and building an internal representation that converts the raw image pixels into a complex understanding 
of the features present within the image. This is also a reason why convolutional neural networks are able to generalize well: they’re able to capture the 
invariances and defining features within classes (e.g. cats vs. dogs) that are agnostic to background noise and other nuisances. Thus, somewhere between where 
the raw image is fed into the model and the output classification label, the model serves as a complex feature extractor. By accessing intermediate layers of 
the model, you're able to describe the content and style of input images.
"""

# Build the model
def vgg_layers(layer_names):
    """Creates a vgg model that returns a list of intermediate output values"""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


## Create the model
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)

## Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("   shape: ", output.numpy().shape)
    print("   min: ", output.numpy().min())
    print("   max: ", output.numpy().max())
    print("   mean: ", output.numpy().mean())
    print()

# Calculate style
"""The content of an image is represented by the values of the intermediate feature maps.
It turns out, the style of an image can be described by the means and correlations across the different feature maps. Calculate a Gram matrix 
that includes this information by taking the outer product of the feature vector with itself at each location, and averaging that outer product 
over all locations. This Gram matrix can be calculated for a particular layer. Can be implemented using the tf.linalg.einsum function:
"""


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


## Extract style content
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print("Styles:")
for name, output in sorted(results["style"].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results["content"].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())


# Run Gradient Descent
style_targets = extractor(style_image)["style"]
content_targets = extractor(content_image)["content"]

image = tf.Variable(content_image)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight = 1e-2
content_weight = 1e4


def style_content_loss(outputs):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# Run a few steps to test
train_step(image)
train_step(image)
train_step(image)
trained_img = tensor_to_image(image)
plt.imshow(trained_img)
plt.title("Trained Image")
plt.show()

# Since it's working, let's perform a longer optimization
import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  plt.imshow(tensor_to_image(image))
  plt.show()
  print("Train step: {}".format(step))
  
end = time.time()
print("Total time: {:.1f}".format(end-start))