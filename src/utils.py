import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from typing import Tuple, List

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