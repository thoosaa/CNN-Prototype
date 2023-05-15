from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras import activations
from keras import backend as K
from tensorflow import keras
import sys

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


def build_model():
    input_shape = (224, 224, 3)
    # 480 x 270 // 640 x 360

    inputs = Input(shape=input_shape)

    conv = Conv2D(96, kernel_size=(11, 11), activation=activations.relu, padding='valid')(inputs)
    conv = MaxPooling2D(pool_size=(3, 3), strides=3)(conv)
    conv = Conv2D(64, kernel_size=(3, 3), activation=activations.relu, padding='valid')(conv)
    conv = MaxPooling2D(pool_size=(3, 3), strides=3)(conv)
    conv = Conv2D(32, kernel_size=(3, 3), activation=activations.relu, padding='valid')(conv)
    conv = MaxPooling2D(pool_size=(3, 3), strides=3)(conv)

    flat = Flatten()(conv)

    keys = Dense(1024, activation=activations.sigmoid)(flat)
    keys = Dropout(0.5)(keys)
    keys = Dense(512, activation=activations.leaky_relu)(keys)
    keys = Dense(256, activation=activations.leaky_relu)(keys)
    keys = Dropout(0.3)(keys)
    keys = Dense(128, activation=activations.leaky_relu)(keys)
    keys = Dense(64, activation=activations.sigmoid)(keys)

    keys_output = Dense(14, activation=activations.sigmoid, name="keys")(keys)

    click = Dense(512, activation=activations.tanh)(flat)
    click = Dropout(0.8)(click)
    click = Dense(512, activation=activations.elu)(click)
    click = Dropout(0.4)(click)
    click = Dense(256, activation=activations.relu)(click)
    click = Dense(256, activation=activations.relu)(click)
    click = Dense(256, activation=activations.relu)(click)
    click = Dropout(0.3)(click)
    click = Dense(128, activation=activations.elu)(click)
    click = Dense(128, activation=activations.tanh)(click)
    click = Dropout(0.1)(click)

    click_output = Dense(5, activation=activations.softmax, name="click")(click)

    xcor = Dense(256, activation=activations.leaky_relu)(flat)
    xcor = Dropout(0.6)(xcor)
    xcor = Dense(256, activation=activations.relu)(xcor)
    xcor = Dropout(0.6)(xcor)
    xcor = Dense(128, activation= activations.leaky_relu)(xcor)
    xcor = Dropout(0.3)(xcor)
    xcor = Dense(128, activation=activations.relu)(xcor)
    xcor = Dropout(0.3)(xcor)
    xcor = Dense(128, activation=activations.leaky_relu)(xcor)
    xcor = Dropout(0.3)(xcor)
    xcor = Dense(64, activation=activations.softmax)(xcor)
    xcor = Dense(64, activation=activations.softmax)(xcor)

    xcor_output = Dense(1, name="x_cor")(xcor)

    ycor = Dense(256, activation=activations.leaky_relu)(flat)
    ycor = Dropout(0.6)(ycor)
    ycor = Dense(256, activation=activations.relu)(ycor)
    ycor = Dropout(0.6)(ycor)
    ycor = Dense(128, activation=activations.leaky_relu)(ycor)
    ycor = Dropout(0.3)(ycor)
    ycor = Dense(128, activation=activations.relu)(ycor)
    ycor = Dropout(0.3)(ycor)
    ycor = Dense(128, activation=activations.leaky_relu)(ycor)
    ycor = Dropout(0.3)(ycor)
    ycor = Dense(64, activation=activations.softmax)(ycor)
    ycor = Dense(64, activation=activations.softmax)(ycor)

    ycor_output = Dense(1, name="y_cor")(ycor)

    mod = Model(inputs=inputs, outputs=[keys_output, click_output, xcor_output, ycor_output])

    return mod


def plot_model(model):
    keras.utils.plot_model(model, "model_2.1.png", show_shapes=True)


def save_model(model):
    mod = model.to_json()
    with open("model_2.1.json", "w") as json_file:
        json_file.write(mod)


model = build_model()
plot_model(model)
save_model(model)
model.summary()


