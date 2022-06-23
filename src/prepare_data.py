from data_preparation.preprocessing import audio_description_information, calculate_respiratory_cycle_lengths, assign_classes, divide_audio_into_respiratory_cycles
from data_preparation.audio_preprocessing import audio_preprocessing
from data_preparation.feature_extraction import load_audio_data, feature_extraction

if __name__ == '__main__':
    dataset_path = './dataset/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/'
    raw_respiratory_cycles_path = 'results/preprocessing/raw_respiratory_cycles/'
    preprocessed_respiratory_cycles_path = 'results/preprocessing/preprocessed_respiratory_cycles/'
    melspectrograms_path = 'results/preprocessing/mel_spectrograms'

    print('Processing data...')
    # Extracting respiratory cycles from audio files based on the annotations included in the dataset
    # Based on the annotations every respiratory cycle is assigned to one of the four classes (normal, wheezes, crackles, both)
    audio_descriptions_df = audio_description_information(dataset_path)
    audio_descriptions_df = calculate_respiratory_cycle_lengths(
        audio_descriptions_df)
    audio_descriptions_df = assign_classes(audio_descriptions_df)

    divide_audio_into_respiratory_cycles(
        dataset_path, raw_respiratory_cycles_path, audio_descriptions_df)

    target_sample_rate = 4000
    # No duration normalization
    target_audio_duration = None

    print('Preprocessing audio files...')
    # Preprocessing raw respiratory cycles. Sample rate normalization
    audio_preprocessing(read_path=raw_respiratory_cycles_path, write_path=preprocessed_respiratory_cycles_path, 
        target_sample_rate=target_sample_rate, normalize_duration=target_audio_duration)

    print('Generating mel-spectrograms from audio files...')
    # Converting preprocessed respiratory cycles into mel spectrograms
    X, y, class_names = load_audio_data(preprocessed_respiratory_cycles_path)
    feature_extraction(melspectrograms_path, X, y, class_names)