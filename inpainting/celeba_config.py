
DATA_PATH = 'data/img_align_celeba'
BATCH_SIZE = 512
NUM_WORKERS = 0
Z_SIZE = 100
MODEL_PATH = 'model/'
CONTINUE_TRAINING = True
DEVICE = 'cuda:0'
ESTIMATOR_DEVICE = 'cuda:0'
label_smoothing = 0.25
k = 1  # how many times to update discriminator for 1 generator update
ENV_NAME = "gan"
NEW_VISDOM_ENV=True
