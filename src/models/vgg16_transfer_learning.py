from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalMaxPooling2D
from keras import Model
import tensorflow as tf
from typing import Tuple, List

def get_vgg_16_model(input_shape: Tuple[int, int] = (73, 99), learning_rate: float = 1e-4) -> Model:
    base_model = VGG16(
        pooling='none',
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape[0], input_shape[1], 3),
    )

    base_model.trainable = False

    x = GlobalMaxPooling2D()(base_model.output)
    outputs = Dense(4, activation = 'softmax')(x)
    vgg16_model = Model(base_model.input, outputs)

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    vgg16_model.compile(optimizer=opt, 
        loss=tf.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'])
    print(vgg16_model.summary())
    return vgg16_model

def unfreeze_vgg16_layers(model: Model, unfreeze_layers: List[str] = ['block4', 'block5'], 
    fine_tune_learning_rate: float = 1e-5) -> Model:
    for layer in model.layers:
        for layer_name in unfreeze_layers:
            if layer.name.startswith(layer_name):
                layer.trainable = True

    opt = tf.optimizers.Adam(learning_rate=fine_tune_learning_rate)
    model.compile(optimizer=opt, 
        loss=tf.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    pass