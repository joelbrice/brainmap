
######## DATA SOURCING ######## ANTONIO
# first set of functions to execute
# anything related to loading input data


# IMPORTS
# load_mri_data
import os
import tensorflow as tf
from colorama import Fore, Style

from brainmap.ml_logic.preprocessor import normalize_mri, one_hot_encode

######## load_mri_data ########
# Description: import data from a local folder into a BatchDataset using tensorflow
# Args: folder path, desired validation split, image and batch size (all set in PARAMS.PY)
# Kwargs: N/A
# Seps: pull data from local folder
#       apply a validation split
#       define the sub dataset name (training vs validation)
#       resizes image to desired size
#       divide the data into a desired batch size for faster handling
# Libraries: tensorflow, colorama

VALIDATION_SPLIT = os.environ.get("VALIDATION_SPLIT")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE")
BATCH_SIZE = os.environ.get("BATCH_SIZE")
SEED = os.environ.get("SEED")


def load_mri_data(train_data_dir):
    """
    Load MRI images from directory and create TensorFlow datasets.
    """
    print(Fore.MAGENTA + "\nLoading MRI data..." + Style.RESET_ALL)

    x_train = tf.keras.utils.image_dataset_from_directory(
            train_data_dir,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
    x_val = tf.keras.utils.image_dataset_from_directory(
            train_data_dir,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
    )

    x_train_normalized = x_train.map(normalize_mri)
    x_train_encoded = x_train_normalized.map(one_hot_encode)

    x_val_normalized = x_val.map(normalize_mri)
    x_val_encoded = x_val_normalized.map(one_hot_encode)


    return x_train_encoded, x_val_encoded
