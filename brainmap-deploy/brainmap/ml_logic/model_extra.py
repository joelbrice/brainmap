
######## DATA SOURCING ######## ANTONIO
# first set of functions to execute
# anything related to loading input data



# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


train_data_dir= "../data/raw_data/Training"
test_data_dir = "../data/raw_data/Testing"
tensorboard_log_dir = './logs'  # Define the log directory for TensorBoard


train_data_dir = "../data/raw_data/Training"
val_data_dir = "../data/raw_data/Validation"

X_train, X_val = load_mri_data(train_data_dir), load_mri_data(val_data_dir)

X_test = load_mri_data(test_data_dir)


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    reg = regularizers.l2(0.001)
    model = Sequential()

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

def compile_model(model: Model, learning_rate=0.0001) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate),

    model.compile(optimizer= optimizer, loss=CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])

    print("✅ Model compiled")

    return model

def train_model(
        model: tf.keras.Model,
        X_train: tf.data.Dataset,
        X_val: tf.data.Dataset,
        batch_size: int = BATCH_SIZE,
        patience: int = 2
    ) -> Tuple[tf.keras.Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
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
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(Fore.GREEN + "✅ Model trained" + Style.RESET_ALL)

    return model, history.history


def evaluate_model(
        model:model,
        X_test: tf.data.Dataset,
        batch_size: int = BATCH_SIZE
    ) -> dict:
    """
    Evaluate trained model performance on the dataset
    """
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
