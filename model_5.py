from tensorflow import keras
from keras import layers
from keras import activations

input_shape = (750, 750, 3)

inputs = keras.Input(shape=input_shape)

x = layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="tanh")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
x = layers.Conv2D(64, kernel_size=(3, 3), padding="valid", activation=activations.leaky_relu)(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = layers.Conv2D(32, kernel_size=(5, 5), padding="valid", activation=activations.leaky_relu)(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=1)(x)
x = layers.Flatten()(x)

x = layers.Dense(512, activation="tanh")(x)
x = layers.Dense(512, activation="tanh")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation=activations.leaky_relu)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation=activations.leaky_relu)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation=activations.leaky_relu)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(128, activation="tanh")(x)
x = layers.Dropout(0.1)(x)

click_type = layers.Dense(10, activation="softmax", name="click_type")(x)
keys = layers.Dense(5, activation="sigmoid", name="keys")(x)
x_coord = layers.Dense(1, name="x_coord")(x)
y_coord = layers.Dense(1, name="y_coord")(x)

model = keras.Model(inputs=inputs, outputs=[click_type, keys, x_coord, y_coord])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss={
        "class": "categorical_crossentropy",
        "label": "binary_crossentropy",
        "number1": "mse",
        "number2": "mse",
    },
    metrics={
        "class": "accuracy",
        "label": "accuracy",
        "number1": "mae",
        "number2": "mae",
    },
)
model.summary()

model = model.to_json()
with open("model_5.json", "w") as json_file:
    json_file.write(model)



