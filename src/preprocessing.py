import os
import pandas as pd
import librosa
import pathlib
import soundfile as sf

def audio_description_information(path: str) -> pd.DataFrame:
    df_list = []
    for filename in os.listdir(path):
        if filename.endswith('txt'):
            annotations_df = pd.read_csv(os.path.join(path, filename), names=[
                'Starts', 'Ends', 'Crackles', 'Wheezes'], delimiter='\t')

            filename, _ = filename.rsplit('.')
            recording_info = filename.split('_')
            recording_info.append(filename)

            columns = ['Patient number', 'Recording index',
                       'Chest location', 'Acquisition mode', 'Recording equipment', 'Filename']

            recording_info_df = pd.DataFrame([recording_info], columns=columns)

            full_audio_description_df = annotations_df.merge(
                recording_info_df, how='cross')

            df_list.append(full_audio_description_df)

    audio_descriptions_df = pd.concat(df_list, ignore_index=True)
    return audio_descriptions_df

def calculate_respiratory_cycle_lengths(audio_descriptions: pd.DataFrame) -> pd.DataFrame:
    audio_descriptions['Respiratory cycle'] = audio_descriptions['Ends'] - \
        audio_descriptions['Starts']
    return audio_descriptions

def assign_classes(audio_descriptions: pd.DataFrame) -> pd.DataFrame:
    audio_descriptions['Class'] = 'normal'
    audio_descriptions.loc[(audio_descriptions['Crackles'] == 1) & (
        audio_descriptions['Wheezes'] == 0), 'Class'] = 'crackles'
    audio_descriptions.loc[(audio_descriptions['Crackles'] == 0) & (
        audio_descriptions['Wheezes'] == 1), 'Class'] = 'wheezes'
    audio_descriptions.loc[(audio_descriptions['Crackles'] == 1) & (
        audio_descriptions['Wheezes'] == 1), 'Class'] = 'both'

    audio_descriptions = audio_descriptions.drop(
        ['Crackles', 'Wheezes'], axis=1)
    return audio_descriptions


def divide_audio_into_respiratory_cycles(read_path: str, write_path: str, audio_descriptions: pd.DataFrame) -> None:
    for filename in os.listdir(read_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(read_path, filename)
            audio, sample_rate = librosa.load(file_path, sr=None)

            filename, _ = filename.rsplit('.')
            annotations = audio_descriptions[audio_descriptions['Filename'] == filename]
            for i, row in annotations.iterrows():
                start = max(0, int(sample_rate * row['Starts']))
                end = min(audio.shape[0], int(
                    sample_rate * row['Ends']))
                respiratory_cycle = audio[start:end]

                respiratory_cycle_filename = '{filename}_{number}.wav'.format(
                    filename=filename, number=i)
                full_write_path = os.path.join(
                    write_path, row['Class'])

                if not pathlib.Path(full_write_path).exists():
                    pathlib.Path(full_write_path).mkdir(
                        parents=True, exist_ok=True)

                full_write_path = os.path.join(
                    full_write_path, respiratory_cycle_filename)

                sf.write(full_write_path, respiratory_cycle, sample_rate)


if __name__ == '__main__':
    data_path = './dataset/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/'
    raw_respiratory_cycles_path = 'results/preprocessing/raw_respiratory_cycles/'

    audio_descriptions_df = audio_description_information(data_path)
    audio_descriptions_df = calculate_respiratory_cycle_lengths(
        audio_descriptions_df)
    audio_descriptions_df = assign_classes(audio_descriptions_df)

    divide_audio_into_respiratory_cycles(
        data_path, raw_respiratory_cycles_path, audio_descriptions_df)