import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, DataFrameIterator
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.backend import clear_session
import gc
from typing import Callable, Tuple, List
from .train import train_model
from .data_augmentation import spec_augmentation
from .utils import plot_confusion_matrix

def prepare_generators(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
    color_mode: str = 'grayscale', input_shape: Tuple[int, int] = (73, 99), 
    batch_size: int = 64) -> Tuple[DataFrameIterator, DataFrameIterator]:

    train_data_generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        preprocessing_function=spec_augmentation,
        )
    val_data_generator = ImageDataGenerator(rescale=1./255)

    train_df = pd.DataFrame({
        'filepaths': X_train,
        'labels': y_train.astype(str)
    })

    test_df = pd.DataFrame({
        'filepaths': X_test,
        'labels': y_test.astype(str)
    })

    train_generator = train_data_generator.flow_from_dataframe(
        dataframe=train_df,
        color_mode=color_mode,
        x_col='filepaths',
        y_col='labels',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='sparse')

    validation_generator = val_data_generator.flow_from_dataframe(
        dataframe=test_df,
        color_mode=color_mode,
        x_col='filepaths',
        y_col='labels',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False)

    return train_generator, validation_generator

def cross_validation_model_evaluation(X: np.ndarray, y: np.ndarray, class_names: List[str], write_path: str, 
    get_model_func: Callable, random_state: int = 123, **kwargs) -> None:

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)

    classification_reports = []
    macro_f1s = []
    y_preds = []
    y_trues = []
    for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        generator_parameters_dict = dict((x, kwargs[x]) 
            for x in ['color_mode', 'input_shape', 'batch_size'] if x in kwargs)

        train_generator, validation_generator = prepare_generators(X_train, y_train, X_test, y_test, 
            **generator_parameters_dict)

        model = get_model_func(**kwargs['model_hyperparameters'])
        model = train_model(train_generator, validation_generator, model, **kwargs)

        y_pred = model.predict(validation_generator)
        y_pred = np.argmax(y_pred, axis=1)
            
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        classification_reports.append(report_df)
        macro_f1s.append(report['macro avg']['f1-score'])
        print(report)

        y_preds.append(y_pred)
        y_trues.append(y_test)

        clear_session()
        gc.collect()

    _write_path, _ = write_path.split('.csv') 
    macro_f1_write_path = _write_path + '_m_f1s.csv'
    report_write_path = _write_path + '_report.csv'

    macro_f1_df = pd.DataFrame(macro_f1s)
    macro_f1_df.to_csv(macro_f1_write_path, mode='w+')

    full_cv_report = pd.concat(classification_reports, join='inner') 
    full_cv_report.to_csv(report_write_path, mode='w+')

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)

    write_path_png = _write_path + '.png'
    plot_confusion_matrix(write_path_png, class_names, y_trues, y_preds)

def simple_split_model_evaluation(X: np.ndarray, y: np.ndarray, class_names: List[str], write_path: str, 
    get_model_func: Callable, test_size: float = 0.2, random_state: int = 123, **kwargs) -> None:
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    generator_parameters_dict = dict((x, kwargs[x]) 
        for x in ['color_mode', 'input_shape', 'batch_size'] if x in kwargs)

    train_generator, validation_generator = prepare_generators(X_train, y_train, X_test, y_test, 
        **generator_parameters_dict)

    model = get_model_func(**kwargs['model_hyperparameters'])
    model = train_model(train_generator, validation_generator, model, **kwargs)

    y_pred = model.predict(validation_generator)
    y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_write_path = write_path + 'report.csv'
    write_path_png = write_path + 'conf_matrix.png'
    write_path_model = write_path + 'saved_model.h5'

    report_df.to_csv(report_write_path, mode='w+')
    plot_confusion_matrix(write_path_png, class_names, y_test, y_pred)
    model.save(write_path_model)