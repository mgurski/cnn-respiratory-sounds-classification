import os
import librosa
import numpy as np
import pathlib
import soundfile as sf

def normalize_audio_length(audio: np.ndarray, sample_rate: int, target_length: int = 3) -> np.ndarray:
    """
    Samples with duration longer than desired are cut after the target length. 
    Samples with duration lower than desired are centered and padded with silence.
    """
    target_sample_length = target_length * sample_rate

    normalized_audio = audio
    if audio.shape[0] > target_sample_length:
        normalized_audio = audio[0:target_sample_length]
    elif audio.shape[0] < target_sample_length:
        normalized_audio = librosa.util.pad_center(
            audio, size=target_sample_length)

    return normalized_audio

def audio_preprocessing(read_path: str, write_path: str, target_sample_rate: int, normalize_duration: None) -> None:
    for label in os.listdir(read_path):
        for filename in os.listdir(os.path.join(read_path, label)):
            file_path = os.path.join(read_path, label, filename)
            audio, _ = librosa.load(file_path, sr=target_sample_rate)

            if normalize_duration:
                audio = normalize_audio_length(
                    audio, target_sample_rate, target_length=normalize_duration)

            label_dir = os.path.join(write_path, label)
            if not pathlib.Path(label_dir).exists():
                pathlib.Path(label_dir).mkdir(
                    parents=True, exist_ok=True)

            full_write_path = os.path.join(label_dir, filename)
            sf.write(full_write_path, audio, target_sample_rate)


if __name__ == '__main__':
    read_path = 'results/preprocessing/raw_respiratory_cycles/'
    write_path = 'results/preprocessing/preprocessed_respiratory_cycles/'

    target_sample_rate = 4000

    #no duration normalization
    target_audio_duration = None

    audio_preprocessing(read_path=read_path, write_path=write_path, 
        target_sample_rate=target_sample_rate, normalize_duration=target_audio_duration)