
######## MODELING ######## GOKHAN/JOEL/ANTONIO
# third set of functions to execute
# anything related to modeling from initializing to training

# IMPORTS
# initialize_model
import tensorflow as tf
from tensorflow.keras import regularizers, models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC

# train_model
import os
from colorama import Fore, Style
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

######## initialize_model ######## JOEL/ANTONIO
# Description: initializes the desired model.
#               Maxpooling to reduce the size of the output of a layer by taking max value of output
#               Batch Normalization to simplify calculations in the following layers by transforming data
#               Dropout layer to speed up calculations by dropping some of the outputs
# Args: image input shape
# Kwargs: N/A
# Seps: Define the type of regularizer and learning rate
#       Initialize a sequential model
#       Input Layer (128x128x1)
#       Block 1 (3x3 Conv) + maxpooling
#       Block 2 (5x5 Conv) + batch normalization + maxpooling
#       Block 3 (3x3 Conv) + batch normalization + maxpooling
#       Block 4 (5x5 Conv) + batch normalization + maxpooling
#       Block 5 (3x3 Conv) + batch normalization
#       Global Average Pooling Layer
#       Dropout Layer
#       Dense Output Layer for Classification
# Libraries: tensorflow, tensorflow.keras
def initialize_model(input_shape: tuple, reg = REGULARIZER) -> Model:

    model = models.Sequential()

    model.add(layers.InputLayer(shape=input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu', kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(4, activation='softmax', kernel_regularizer=reg))

    print("✅ Model initialized")

    return model

######## compile_model ######## JOEL/ANTONIO
# Description: compile the model according to the desired metrics
# Args: model and optimizer's learning rate
# Kwargs: N/A
# Seps: define the optimizer and its appropriate learning rate\
#       define the loss function
#       compile the model with desired parameters
# Libraries: tensorflow, tensorflow.keras.optimizers, tensorflow.keras.losses, tensorflow.keras.metrics
def compile_model(model: Model, learning_rate = OPTIMIZER_LEARNING_RATE) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate)
    loss = CategoricalCrossentropy(from_logits=False)

    model.compile(optimizer= optimizer, loss=loss,
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])

    print("✅ Model compiled")

    return model

######## train_model ######## JOEL/ANTONIO
# Description: train the model on the desired dataset
# Args: model to use, train and validation datasets for the training, batch size and patience
# Kwargs: N/A
# Seps: define checkpoint directory if not already existing
#       define callbacks parameters:
#           checkpoint to save model with minimum validation loss
#           early stopping to stop training when validation loss stops decreasing
#           learning rate reducer when the validation loss plateaus
#       train the model with callbacks and storing the training history
#       return a tuple with the fitted model and the history of results
# Libraries: tensorflow, colorama, os, tensorflow.keras.callbacks
def train_model(
        model: tf.keras.Model,
        X_train: tf.data.Dataset,
        X_val: tf.data.Dataset,
        batch_size: int = BATCH_SIZE,
        patience: int = PATIENCE
        epochs = EPOCHS
    ) -> Tuple[tf.keras.Model, dict]:

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    checkpoint_dir = './checkpoints'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks = [
        ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
                        monitor='val_loss', mode='min', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train,
        validation_data=X_val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(Fore.GREEN + "✅ Model trained" + Style.RESET_ALL)

    return model, history.history

######## evaluate_model ######## JOEL/ANTONIO
# Description: evaluate trained model performance on the dataset. returns a dictionary with the metrics
# Args: model to use and test dataset for comparison
# Kwargs: N/A
# Seps: apply evaluate method to the trained model, if it exists. return an error if it does not
# Libraries: tensorflow
def evaluate_model(
        model:model,
        X_test: tf.data.Dataset,
        batch_size: int = BATCH_SIZE
    ) -> dict:

    print(Fore.BLUE + f"\nEvaluating model..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        X_test,
        batch_size=batch_size,
        verbose=1
    )

    print(Fore.GREEN + "✅ Model evaluated" + Style.RESET_ALL)

    return dict(zip(model.metrics_names, metrics))
