
DATA_PATH = 'data/img_align_celeba'
BATCH_SIZE = 512
NUM_WORKERS = 0
Z_SIZE = 100
MODEL_PATH = 'conditional_model/'
CONTINUE_TRAINING = False
CUDA_DEVICE = 1
conditions = ['Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat']
Y_SIZE = len(conditions)
label_smoothing = 0.25
k = 1  # how many times to update discriminator for 1 generator update
ENV_NAME = "conditional_gan"
