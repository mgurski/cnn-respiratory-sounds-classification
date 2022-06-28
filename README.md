# CNN based respiratory sounds classification
Application of convolutional neural networks to the classification of respiratory sounds. Three cnn models (custom model, vgg16, resnet50v2) are trained on the [The ICHBI2017 Dataset [1]](#1). 

Mel-spectrograms were used as the input representation.

<img src="results/random_samples.png" alt="random_samples" width="320" height="200"/>

</br>

Time and frequency masking as in [SpecAugment [2]](#2) and horizontal flips were used for online data augmentation.

<img src="results/augmentations.png" alt="aug" width="320" height="200"/>

</br>

The best results were obtained for the custom model which used non-standard convolutional filters. Normalized confusion matrix obtained for the custom model is shown below. A simple train/test split was used with 80% of the data used for training and 20% used for validation.

<img src="results/custom_model/conf_matrix.png" alt="conf" width="320" height="200"/>

</br>

## Technologies
Main technologies used in the project:
* Python 3.9.0
* Keras & TensorFlow 2.9
* Librosa 0.9.1

## Setup
Create a virtual environment and install requirements \
`python3 -m venv venv/` \
`source venv/bin/activate` \
`pip3 install -r requirements.txt` 

Run the prepare_data script. The downloaded data should be put in the project root in a folder named dataset \
`python3 src/prepare_data.py` 

Run the evaluation of the chosen model \
`python3 src/main.py [-h] --model {custom, vgg16, resnet50v2} [-p P] [--test_size TEST_SIZE] [--random_state RANDOM_STATE]`

# References
<a href="https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge" id="1">
1. Rocha BM et al. (2019) "An open access database for the evaluation of respiratory sound classification algorithms" Physiological Measurement 40 035001
</a>

<a href="https://arxiv.org/abs/1904.08779" id="2">
2. Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le. "SpecAugment: A simple data augmentation method for automatic speech recognition." In Interspeech 2019. ISCA, sep 2019
</a>
