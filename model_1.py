from tensorflow import keras
from keras import layers

input_shape = (750, 750, 3)

inputs = keras.Input(shape=input_shape)

x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

click_type = layers.Dense(10, activation="softmax", name="click_type")(x)
keys = layers.Dense(5, activation="sigmoid", name="keys")(x)
x_coord = layers.Dense(1, name="x_coord")(x)
y_coord = layers.Dense(1, name="y_coord")(x)

model = keras.Model(inputs=inputs, outputs=[click_type, keys, x_coord, y_coord])

model.summary()

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


model = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model)





