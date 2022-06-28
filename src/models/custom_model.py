import tensorflow as tf
from typing import Tuple
from keras import Model

def get_custom_model(input_shape: Tuple[int, int] = (73, 99), learning_rate: float = 0.0001, 
    spatial_dropout: float = 0.1, filter_sizes: Tuple[int, int, int] = (64, 128, 256), 
    kernel_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((11, 7), (9, 5), (5, 5))) -> Model:
    """
    Custom model with non-standard convolutional filter sizes. The idea is based on the differences between spectrograms 
    and standard images - mainly the non-local frequency components distribution on the y axis. 
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1)))

    # conv layers
    # block 0
    model.add(tf.keras.layers.Conv2D(filters=filter_sizes[0], kernel_size=kernel_sizes[0], padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(padding='same'))
    model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout))

    # block 1
    model.add(tf.keras.layers.Conv2D(filters=filter_sizes[1], kernel_size=kernel_sizes[1], padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=filter_sizes[1], kernel_size=kernel_sizes[1], padding='same', activation='relu', 
        strides=(2, 2)))
    model.add(tf.keras.layers.MaxPool2D(padding='same'))
    model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout))

    # block 2
    model.add(tf.keras.layers.Conv2D(filters=filter_sizes[2], kernel_size=kernel_sizes[2], padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=filter_sizes[2], kernel_size=kernel_sizes[2], padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=filter_sizes[2], kernel_size=kernel_sizes[2], padding='same', activation='relu', 
        strides=(2, 2)))
    model.add(tf.keras.layers.MaxPool2D(padding='same'))
    model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout))

    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.Dense(4, activation = 'softmax'))

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    print(model.summary())
    return model