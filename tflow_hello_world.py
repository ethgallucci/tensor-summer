import tensorflow as tf

print("Tensorflow version: ", tf.__version__)

# Load a dataset: MNIST handwritten digits
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a Machine Learning model by stacking keras layers
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

predictions = model(x_train[:1]).numpy()
print(predictions)

# Converts these logits or 'log-odds' to probabilities for each class
probs = tf.nn.softmax(predictions).numpy()

# Define a loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Configure and compile the model
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train and evaluate the model
model.fit(x_train, y_train, epochs=5)  # Adjust model params and minimize the loss
model.evaluate(
    x_test, y_test, verbose=2
)  # Checks the model for performance, usually on a 'validation-set' or 'test-set'

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])
