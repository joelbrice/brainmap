import tensorflow as tf

def normalize_mri(image):
    """Normalize an MRI image tensor."""
    mean, std = tf.reduce_mean(image), tf.math.reduce_std(image)
    return (image - mean) / (std + tf.keras.backend.epsilon())
