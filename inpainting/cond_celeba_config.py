
class Config(dict):
    def __init__(self):
        super().__init__()
        self.__dict__ = self
        self.DATA_PATH = 'data/img_align_celeba'
        self.BATCH_SIZE = 512
        self.NUM_WORKERS = 0
        self.Z_SIZE = 100
        self.MODEL_PATH = 'conditional_model/'
        self.CONTINUE_TRAINING = False
        self.DEVICE = 'cuda:1'
        self.ESTIMATOR_DEVICE = 'cuda:1'

        self.conditions = ['Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat']
        self.Y_SIZE = len(self.conditions)
        self.label_smoothing = 0.25
        self.k = 1  # how many times to update discriminator for 1 generator update
        self.ENV_NAME = "conditional_gan"
        self.NEW_VISDOM_ENV=True
