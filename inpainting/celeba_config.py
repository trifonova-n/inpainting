from gan.hyperparameters import GeneratorParams, DiscriminatorParams, TrainingParams


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__ = self
        self.DATA_PATH = 'data/img_align_celeba'
        self.BATCH_SIZE = 512
        self.NUM_WORKERS = 0
        self.Z_SIZE = 100
        self.MODEL_PATH = 'model/'
        self.CONTINUE_TRAINING = False
        self.DEVICE = 'cuda:0'
        self.ESTIMATOR_DEVICE = 'cuda:0'
        self.label_smoothing = 0.25
        self.k = 1  # how many times to update discriminator for 1 generator update
        self.ENV_NAME = "gan"
        self.NEW_VISDOM_ENV=True
        for key in kwargs:
            self[key] = kwargs[key]
        self.generator_params = GeneratorParams(**kwargs.get('generator_params', {}))
        self.discriminator_params = DiscriminatorParams(**kwargs.get('discriminator_params', {}))
        self.training_params = TrainingParams(**kwargs.get('training_params', {}))

