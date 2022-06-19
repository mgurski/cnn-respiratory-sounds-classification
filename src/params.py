import json
import pathlib

def write_parameters(dict: dict, path: str) -> None:
    json_string = json.dumps(dict, indent=2)
    with open(path, mode='w+') as f:
        f.write(json_string) 

def load_parameters(path: str) -> dict:
    with open(path, mode='r') as f:
        dict = json.load(f)
    return dict

DEFAULT_INPUT_REPRESENTATION_HYPERPARAMETERS = {
    "sample_rate": 4000,
    "normalize_duration": None,
    "n_fft": 256,
    "n_mels": 64, 
}

DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS = {
    "model_hyperparameters": {
        "input_shape": [73, 99],
        "learning_rate": 0.0001,
        "spatial_dropout": 0.1,
        "filter_sizes": [64, 128, 256],
        "kernel_sizes": [ [11, 9], [9, 7], [5, 5] ],
        },
    "color_mode": 'grayscale',
    "batch_size": 64,
    "epochs": 150,
}

DEFAULT_VGG16_HYPERPARAMETERS = {
    "model_hyperparameters": {
        "input_shape": [73, 99],
        "learning_rate": 0.0001,
    },
    "fine_tune_hyperparameters": {
        "fine_tune_learning_rate": 0.00001,
        "unfreeze_layers": ['block4', 'block5']
    },
    "fine_tune_epochs": 150,
    "color_mode": 'rgb',
    "epochs": 100,
    "batch_size": 64, 
}

DEFAULT_RESNET50V2_HYPERPARAMETERES = {
    "model_hyperparameters": {
        "input_shape": [73, 99],
        "learning_rate": 0.0001,
    },
    "fine_tune_hyperparameters": {
        "fine_tune_learning_rate": 0.000005,
        "unfreeze_layers": ['conv4', 'conv5']
    },
    "fine_tune_epochs": 150,
    "color_mode": 'rgb',
    "epochs": 100,
    "batch_size": 64, 
}

if __name__ == '__main__':
    
    params_dir_path = 'results/params/'
    if not pathlib.Path(params_dir_path).exists():
        pathlib.Path(params_dir_path).mkdir(parents=True, exist_ok=True)

    write_parameters(DEFAULT_INPUT_REPRESENTATION_HYPERPARAMETERS, params_dir_path + 'default_input_representation_hyperparameters.json')

    write_parameters(DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS, params_dir_path + 'default_custom_model_hyperparameters.json')
        
    write_parameters(DEFAULT_VGG16_HYPERPARAMETERS, params_dir_path + 'default_vgg16_hyperparameters.json')

    write_parameters(DEFAULT_RESNET50V2_HYPERPARAMETERES, params_dir_path + 'default_resnet50v2_hyperparameters.json')

    kernel_size_hyperparameters = {
        "kernel_sizes": [
            [[3, 3], [3, 3], [3, 3]],
            [[7, 7], [5, 5], [3, 3]],
            [[9, 7], [7, 5], [3, 3]],
            [[11, 9], [9, 7], [5, 5]],
            [[11, 11], [9, 9], [5, 5]]
        ]

    }

    write_parameters(kernel_size_hyperparameters, params_dir_path + 'kernel_size_hyperparameters_analysis.json')
