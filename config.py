import yaml

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

ROOT_DIR = cfg['ROOT_DIR']
GPU = cfg['GPU']
MODEL_STATE_FILE = cfg['MODEL_STATE_FILE']

BATCH_SIZE = cfg['TRAINING_INFO']['BATCH_SIZE']
MODEL_TYPE = cfg['TRAINING_INFO']['MODEL_TYPE']
LR = cfg['TRAINING_INFO']['LR']
EPOCHS = cfg['TRAINING_INFO']['EPOCHS']
MODEL_ARGS = cfg['TRAINING_INFO']['MODEL_ARGS']