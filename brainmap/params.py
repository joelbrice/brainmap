import os

############### DATA.PY ############### ANTONIO
TRAIN_DATA_DIR = "../data/raw_data/Training"
TEST_DATA_DIR = "../data/raw_data/Testing"

### load_mri_data###
BATCH_SIZE = 16
DATA_SIZE = os.environ.get("DATA_SIZE")
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (128,128)
SEED = 42
TABLE = f'processed_{DATA_SIZE}'

############### MODEL.PY ############### ANTONIO
### initialize_model ###
REGULARIZER = regularizers.l2(0.001)

### compile_model ###
OPTIMIZER_LEARNING_RATE = 0.0001

### train_model ##
BATCH_SIZE = 16
PATIENCE = 1
EPOCHS = 500



DATA_SIZE = os.environ.get("DATA_SIZE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

MODEL_TARGET = os.environ.get("MODEL_TARGET")

SERVICE_URL = os.environ.get("SERVICE_URL")

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".joelbrice", "brainmap", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".joelbrice", "brainmap", "training_outputs")
