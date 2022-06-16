from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Dense, GlobalMaxPooling2D
from keras import Model
import tensorflow as tf
from typing import Tuple, List

def get_resnet50v2_model(input_shape: Tuple[int, int] = (73, 99), learning_rate: float = 1e-4) -> Model:
    base_model = ResNet50V2(
        pooling='none',
        include_top=False,
        weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3),
    )

    base_model.trainable = False

    x = GlobalMaxPooling2D()(base_model.output)
    outputs = Dense(4, activation = 'softmax')(x)
    model = Model(base_model.input, outputs)

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, 
        loss=tf.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'])
    print(model.summary())
    return model

def unfreeze_resnet50v2_layers(model: Model, unfreeze_layers: List[str] = ['conv2', 'conv3', 'conv4', 'conv5'], 
    learning_rate: float = 1e-5) -> Model:
    for layer in model.layers:
        for layer_name in unfreeze_layers:
            if layer_name in layer.name:
                layer.trainable = True

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, 
        loss=tf.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    pass