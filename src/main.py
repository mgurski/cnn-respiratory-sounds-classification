import pathlib
import argparse
from training.evaluation import simple_split_model_evaluation
from models.custom_model import get_custom_model
from models.resnet50v2_transfer_learning import get_resnet50v2_model, unfreeze_resnet50v2_layers
from models.vgg16_transfer_learning import get_vgg_16_model, unfreeze_vgg16_layers
from training.params import DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS, DEFAULT_RESNET50V2_HYPERPARAMETERES, DEFAULT_VGG16_HYPERPARAMETERS
from utilities.utils import check_valid_model_hyperparameters_json_path, load_data

if __name__ == '__main__':
    data_path = 'results/preprocessing/mel_spectrograms'

    parser = argparse.ArgumentParser(description='Model simple split evaluation')
    parser.add_argument('--model', help='Choose the model to be evaluated. The options are custom, vgg16 and resnet50v2', 
        choices=['custom', 'vgg16', 'resnet50v2'], required=True)
    parser.add_argument('-p', 
        help='Path to a json file with hyperparameters to start the model with a different hyperparameter set than the default')
    parser.add_argument('--test_size', help='Validation split size', type=float, default=0.2)
    parser.add_argument('--random_state', help='Random state', type=int, default=123)
    args = parser.parse_args()
    
    if args.model == 'custom':
        write_path = 'results/custom_model/'
        if not pathlib.Path(write_path).exists():
            pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)

        custom_model_hyperparameters = DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS
        if args.p:
            custom_model_hyperparameters = check_valid_model_hyperparameters_json_path(parser, 
                DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS, args.p)

        X, y, class_names = load_data(data_path, args.random_state)
        simple_split_model_evaluation(X, y, class_names, write_path, get_custom_model, args.test_size, args.random_state,
            **custom_model_hyperparameters)

    if args.model == 'vgg16':
        write_path = 'results/vgg16/'
        if not pathlib.Path(write_path).exists():
            pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)

        vgg16_hyperparameters = DEFAULT_VGG16_HYPERPARAMETERS
        if args.p:
            vgg16_hyperparameters = check_valid_model_hyperparameters_json_path(parser, 
                DEFAULT_VGG16_HYPERPARAMETERS, args.p)

        X, y, class_names = load_data(data_path, args.random_state)
        simple_split_model_evaluation(X, y, class_names, write_path, get_vgg_16_model, args.test_size, args.random_state,
            **vgg16_hyperparameters, unfreeze_layers_func=unfreeze_vgg16_layers)

    if args.model == 'resnet50v2':
        write_path = 'results/resnet50v2/'
        if not pathlib.Path(write_path).exists():
            pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)

        resnet50v2_hyperparameters = DEFAULT_RESNET50V2_HYPERPARAMETERES
        if args.p:
            resnet50v2_hyperparameters = check_valid_model_hyperparameters_json_path(parser, 
                DEFAULT_RESNET50V2_HYPERPARAMETERES, args.p)

        X, y, class_names = load_data(data_path, args.random_state)
        simple_split_model_evaluation(X, y, class_names, write_path, get_resnet50v2_model, args.test_size, args.random_state,
            **resnet50v2_hyperparameters, unfreeze_layers_func=unfreeze_resnet50v2_layers)