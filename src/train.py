from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_preprocessing.image import DataFrameIterator
from typing import Callable
from keras import Model

def train_model(train_generator: DataFrameIterator, validation_generator: DataFrameIterator, model: Model, 
    epochs: int = 60, unfreeze_layers_func: Callable = None, **kwargs) -> Model:
    es_callback = EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            verbose=0,
            mode='auto',
            restore_best_weights=True)

    checkpoint_filepath = 'results/tmp/checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(x=train_generator, validation_data=validation_generator, epochs=epochs, 
        callbacks=[es_callback, model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)

    if unfreeze_layers_func is not None:
        model = unfreeze_layers_func(model, **kwargs['fine_tune_hyperparameters'])

        train_generator.reset()
        validation_generator.reset()

        fine_tune_history = model.fit(x=train_generator,
                    validation_data=validation_generator, 
                    callbacks=[es_callback, model_checkpoint_callback], 
                    epochs=kwargs['fine_tune_epochs'])

        model.load_weights(checkpoint_filepath)

    return model