import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from typing import Tuple, List
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from .params import load_parameters

def filepaths_to_df(dir_path: str) -> pd.DataFrame:
    filepaths = []
    labels = []
    for label in os.listdir(dir_path):
        for filename in os.listdir(os.path.join(dir_path, label)):
            file_path = os.path.join(dir_path, label, filename)
            filepaths.append(file_path)
            labels.append(label)

    filepaths_df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    filepaths_df['labels'] = filepaths_df['labels'].astype('category')
    return filepaths_df

def load_data(read_path: str, shuffle_random_state: int = 123) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    filepaths_df = filepaths_to_df(read_path)
    filepaths_df = shuffle(filepaths_df, random_state=shuffle_random_state)
    filepaths_df.reset_index(inplace=True, drop=True)

    class_names = filepaths_df.labels.cat.categories.tolist()
    filepaths_df.labels = filepaths_df.labels.cat.codes

    X = np.array(filepaths_df.filepaths)
    y = np.array(filepaths_df.labels.astype(int))

    return X, y, class_names

def plot_confusion_matrix(save_path: str, class_names: List[str], y_test: np.ndarray, y_pred: np.ndarray) -> None:
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(4), normalize='true')
    df = pd.DataFrame(conf_matrix, 
        index = [i for i in class_names], columns = [i for i in class_names])
    plt.figure(figsize = (10, 7))
    ax = sns.heatmap(df, annot=True, annot_kws={"size": 12}, fmt='.2g', cmap='Blues')
    ax.set(xlabel='Predicited', ylabel='True')
    plt.savefig(save_path) 

def sub_dict_params(dict_a: dict, dict_b: dict) -> dict:
    for key in dict_a:
        if key in dict_b:
            dict_a[key] = dict_b[key]
    return dict_a

def check_image_shape(image_path: str) -> Tuple[int, int]:
    img = plt.imread(image_path)
    img_shape = img.shape
    return (img_shape[0], img_shape[1])

def check_valid_model_hyperparameters_json_path(parser: argparse.ArgumentParser, default_hyperparameters_dict: dict, 
    path: str) -> bool:
    if path.endswith('.json'):
        json_dict = load_parameters(path)
        if default_hyperparameters_dict.keys() == json_dict.keys():
            return json_dict
    parser.error('Not a valid hyperparameters json path for this model')