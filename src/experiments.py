from params import load_parameters, DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS, DEFAULT_RESNET50V2_HYPERPARAMETERES, DEFAULT_VGG16_HYPERPARAMETERS
from models.custom_model import get_custom_model
from utils import sub_dict_params, load_data
from sklearn.model_selection import ParameterGrid
from evaluation import cross_validation_model_evaluation
from models.vgg16_transfer_learning import get_vgg_16_model, unfreeze_vgg16_layers
from models.resnet50v2_transfer_learning import get_resnet50v2_model, unfreeze_resnet50v2_layers
import pathlib

def model_hyperparameters_analysis(image_path: str, write_results_path: str, 
    custom_model_hyperparameters_dict: dict, test_hyperparameters_dict: dict) -> None:

    # load image filepaths
    X, y, target_names = load_data(image_path)

    # model evaluation for different hyperparameters
    param_grid = ParameterGrid(test_hyperparameters_dict)
    for _dict in param_grid:
        model_params = sub_dict_params(custom_model_hyperparameters_dict, _dict)

        keys = "_".join(list(_dict.keys()))
        value_list = [ str(value) for value in _dict.values() ]
        values = "_".join(value_list)

        current_test_write_path = write_results_path + values + '.csv'
        cross_validation_model_evaluation(X, y, target_names, current_test_write_path, get_custom_model, **model_params)

def model_comparison(image_path, write_results_dir):
    X, y, target_names = load_data(image_path)        
 
    # same random state to compare all of the methods on the same data splits and the same sampling seed
    random_state = 123

    # evaluate custom model
    write_path = write_results_dir + '/custom_model.csv'
    custom_model_params = DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS
    
    cross_validation_model_evaluation(X, y, target_names, write_path, get_custom_model, random_state=random_state,
        **custom_model_params)

    # evaluate vgg16 model
    write_path = write_results_dir + '/vgg16_model.csv'
    vgg16_model_params = DEFAULT_VGG16_HYPERPARAMETERS

    cross_validation_model_evaluation(X, y, target_names, write_path, get_vgg_16_model, unfreeze_layers_func=unfreeze_vgg16_layers,
        random_state=random_state, **vgg16_model_params)

    # evaluate resnet50v2 model
    write_path = write_results_dir + '/resnet50v2_model.csv'
    resnet50v2_model_params = DEFAULT_RESNET50V2_HYPERPARAMETERES

    cross_validation_model_evaluation(X, y, target_names, write_path, get_resnet50v2_model, 
    unfreeze_layers_func=unfreeze_resnet50v2_layers, random_state=random_state, **resnet50v2_model_params)

if __name__ == '__main__':
    raw_wav_path = 'results/preprocessing/raw_respiratory_cycles/'
    preprocessed_wav_path = 'results/preprocessing/preprocessed_respiratory_cycles/'
    image_path = 'results/preprocessing/mel_spectrograms/'

    # kernel size
    write_results_dir = 'results/experiments/conv_layers_hyperparameters/'
    if not pathlib.Path(write_results_dir).exists():
        pathlib.Path(write_results_dir).mkdir(parents=True, exist_ok=True)

    test_hyperparameters_dict = load_parameters('results/params/kernel_size_hyperparameters_analysis.json')
    custom_model_hyperparameters_dict = DEFAULT_CUSTOM_MODEL_HYPERPARAMETERS
    model_hyperparameters_analysis(image_path, write_results_dir, 
        custom_model_hyperparameters_dict, test_hyperparameters_dict)

    # compare models
    write_results_dir = 'results/experiments/model_comparison/'
    if not pathlib.Path(write_results_dir).exists():
        pathlib.Path(write_results_dir).mkdir(parents=True, exist_ok=True)

    model_comparison(image_path, write_results_dir)