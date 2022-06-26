from librosa import display
import librosa
import shutil
import numpy as np
from typing import List, Tuple
import pathlib
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import random

def load_audio_data(dir_path: str, target_sample_rate: int = 4000, 
    shuffle_random_state: int = 123) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = []
    y = []
    class_names = []
    for i, label in enumerate(os.listdir(dir_path)):
        class_names.append(label)
        for filename in os.listdir(os.path.join(dir_path, label)):
            if filename.endswith('.wav'):
                full_read_path = os.path.join(dir_path, label, filename)

                audio, _ = librosa.load(
                    full_read_path, sr=target_sample_rate)

                audio = librosa.util.normalize(audio)

                X.append(audio)
                y.append(i)

    X = np.array(X, dtype=object)
    y = np.array(y)

    X, y = shuffle(X, y, random_state=shuffle_random_state)
    return X, y, class_names

def mel_spec(audio_samples: np.ndarray, sample_rate: int = 4000, n_fft: int = 256, n_mels: int = 64) -> np.ndarray:
    ms = librosa.feature.melspectrogram(y=audio_samples, sr=sample_rate, n_fft=n_fft, 
        hop_length=int( n_fft / 2), n_mels=n_mels)
    ms_dB = librosa.power_to_db(ms, ref=np.max)
    return ms_dB

def feature_extraction(write_path: str, X: np.ndarray, y: np.ndarray, class_names: List[str],
    sample_rate: int = 4000, n_fft: int = 256, n_mels: int = 64, dpi: int = 20) -> None:
    try:
        shutil.rmtree(write_path)
    except OSError as e:
        print("Error: %s : %s" % (write_path, e.strerror))

    for i, audio_samples in enumerate(X):
        label_dir = os.path.join(write_path, class_names[y[i]])        
        
        if not pathlib.Path(label_dir).exists():
            pathlib.Path(label_dir).mkdir(
                parents=True, exist_ok=True)

        audio_samples = np.array(audio_samples, dtype=float)
        ms = mel_spec(audio_samples, sample_rate, n_fft, n_mels)
        display.specshow(ms, y_axis='mel', x_axis='time', sr=sample_rate)
        plt.axis('off')
        full_write_path = os.path.join(label_dir, 'spectrogram_{}.png'.format(i))

        plt.axis('off')
        plt.savefig(full_write_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.cla()
        plt.clf()

def display_random_samples(X: np.ndarray, y: np.ndarray, class_names: List[str]) -> None:
    random.seed(1234)
    random_index = random.sample(range(len(y)), 2)

    X, y = X[random_index], y[random_index]

    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    img = display.waveshow(X[0], sr=4000, ax=ax[0, 0])
    ax[0, 0].set(title='{}'.format(class_names[y[0]]))

    ms = mel_spec(X[0])
    img = display.specshow(ms, y_axis='mel', x_axis='time', sr=4000, hop_length=128, ax=ax[0, 1])
    fig.colorbar(img, ax=ax[0, 1], format='%+2.0f dB')
    ax[0, 1].set(title='{}'.format(class_names[y[0]]))

    img = display.waveshow(X[1], sr=4000, ax=ax[1, 0])
    ax[1, 0].set(title='{}'.format(class_names[y[1]]))

    ms = mel_spec(X[1])
    img = display.specshow(ms, y_axis='mel', x_axis='time', sr=4000, hop_length=128, ax=ax[1, 1])
    fig.colorbar(img, ax=ax[1, 1], format='%+2.0f dB')
    ax[1, 1].set(title='{}'.format(class_names[y[1]]))

    plt.tight_layout()
    plt.savefig('results/random_samples.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    read_path = 'results/preprocessing/preprocessed_respiratory_cycles'
    write_path = 'results/preprocessing/mel_spectrograms'

    X, y, class_names = load_audio_data(read_path)
    feature_extraction(write_path, X, y, class_names)
    #display_random_samples(X, y, class_names)
