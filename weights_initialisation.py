import tensorflow as tf
import json


def initialize_model_weights(filepath):
    with open(filepath, 'r') as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            if layer.activation.__name__ in ['sigmoid', 'tanh']:
                initializer = tf.keras.initializers.glorot_uniform()
            elif layer.activation.__name__ in ['relu', 'elu', 'leaky_relu']:
                initializer = tf.keras.initializers.he_uniform()
            else:
                initializer = tf.keras.initializers.glorot_uniform()
            layer.kernel_initializer = initializer

    return model


model = initialize_model_weights('model_6.json')
model.save_weights('model_6.h5')
