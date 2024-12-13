
######## DATA PREPROCESSING ######## MARKUS/JOEL/ANTONIO
# second set of functions to execute
# anything related to manipulating input data before modeling

# IMPORTS
# normalize_mri
import tensorflow as tf

######## normalize_mri ######## MARKUS/JOEL/ANTONIO
# ----> USE .MAP METHOD TO APPLY TO BATCHDATASET <----
# Description: normalizes the MRI images
# Args: dataset to operate on
# Kwargs: label does not need to be specified, the function works with .map method.
#         needs to be returned by the function to keep the dataset format
# Seps: calculates mean and standard deviation of the dataset
#       normalizes dataset through Z-scores (measure of stdev from the mean)
# Libraries: tensorflow, colorama, tensorflow.keras.utils
def normalize_mri(dataset, label):
    mean = tf.reduce_mean(dataset)
    std = tf.math.reduce_std(dataset)
    temp = (dataset - mean) / (std + tf.keras.backend.epsilon())
                                    # tf.keras.backend.epsilon() returns a small constant value used to prevent
                                    # numerical instability or division by zero in calculations
    normalized_dataset = tf.image.adjust_contrast(temp, 2)

    return normalized_dataset, label


######## one_hot_encode ######## ANTONIO
# ----> USE .MAP METHOD TO APPLY TO BATCHDATASET <----
# Description: encodes the label of each image in the tensor dataset
# Args: dataset to operate on and the classes (label) to one-hot-encode
# Kwargs: label does not need to be specified, the function works with .map method
#         needs to be returned by the function to keep the dataset format
# Seps: obtains a list of the classes in the dataset
#       associates a numerical value to the class of an image
# Requirements: tensorflow
def one_hot_encode(dataset, label):
    # Assuming 'num_classes' is defined
    class_names = dataset.class_names
    return dataset, tf.one_hot(label, depth=len(class_names))
