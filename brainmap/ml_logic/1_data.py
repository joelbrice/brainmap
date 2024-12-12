
######## DATA SOURCING ######## ANTONIO
# first set of functions to execute
# anything related to loading input data


# IMPORTS
# load_mri_data
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from colorama import Fore, Style

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
def load_mri_data(train_data_dir, validation_split=VALIDATION_SPLIT, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED):
    """
    Load MRI images from directory and create TensorFlow datasets.
    """
    print(Fore.MAGENTA + "\nLoading MRI data..." + Style.RESET_ALL)

    train_ds = image_dataset_from_directory(
            train_data_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )
    val_ds = image_dataset_from_directory(
            train_data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
    )

    return train_ds, val_ds
