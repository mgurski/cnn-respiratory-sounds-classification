import numpy as np
import matplotlib.pyplot as plt
import math
from keras_preprocessing.image import ImageDataGenerator, load_img
import random
from utilities.utils import load_data

def time_mask(image: np.ndarray, num: int = 1, min: float = 0.05, max: float = 0.2, 
    p: int = 1) -> np.ndarray:
    image_width = image.shape[1]
    for i in range(num):
        if np.random.uniform(0, 1) < p:
            mask_width = np.random.uniform(min, max)
            mask_width = math.ceil(mask_width * image_width)
            start = np.random.randint(0, (image_width - mask_width))
            image[:, start:start+mask_width] = 0
    return image

def frequency_mask(image: np.ndarray, num: int = 1, min: float = 0.05, max: float = 0.2, 
    p: int = 1) -> np.ndarray:
    image_height = image.shape[0]
    for i in range(num):
        if np.random.uniform(0, 1) < p:
            mask_height = np.random.uniform(min, max)
            mask_height = math.ceil(mask_height * image_height)
            start = np.random.randint(0, (image_height - mask_height))
            image[start:start+mask_height, :] = 0
    return image

def spec_augmentation(image: np.ndarray, num: int = 1, min: float = 0.05, max: float = 0.2, 
    p: int = 1) -> np.ndarray:
    """
    Frequency and time masking based on SpecAugment
    """
    image = time_mask(image, num, min, max, p)
    image = frequency_mask(image, num, min, max, p)
    return image

def test_augmentations(X: np.ndarray, y: np.ndarray) -> None:
    random.seed(1234)
    random_index = random.sample(range(len(y)), 1)
    X, y = X[random_index], y[random_index]

    fig, ax = plt.subplots(3, 3)

    data_generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        preprocessing_function=spec_augmentation)

    img = load_img(X[0], color_mode='grayscale')
    img = np.array(img)
    img = img[..., np.newaxis]
    img = img[np.newaxis, ...]
    iter = data_generator.flow(img, batch_size=1)

    for i in range(3):
        for j in range(3):
            image = next(iter)[0]
            ax[i, j].imshow(image, cmap='magma')
            ax[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/augmentations.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    read_path = 'results/preprocessing/mel_spectrograms/'
    X, y, _ = load_data(read_path)
    test_augmentations(X, y)

