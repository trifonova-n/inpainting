

class Params(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class GeneratorParams(Params):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_shape = self.get('min_shape', (1024, 4, 4))
        self.channel_scaling_factor = self.get('channel_scaling_factor', 2)
        self.out_shape = self.get('out_shape', (3, 64, 64))
        # we don't add batchnorm layers in several first
        # and end layers
        # index of first layer with batchnorm
        self.bn_start_idx = self.get('bn_start_idx', 0)
        # index of last layer to have batchnorm
        self.bn_end_idx = self.get('bn_end_idx', -3)
        self.z_size = self.get('z_size', 100)


class DiscriminatorParams(Params):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = self.get('input_shape', (3, 64, 64))
        self.start_channels = self.get('start_channels', 32)
        self.channel_scaling_factor = self.get('channel_scaling_factor', 2)
        self.feature_img_size = self.get('feature_img_size', 2)
        # we don't add batchnorm layers in several first
        # and end layers
        # index of first layer with batchnorm
        self.bn_start_idx = self.get('bn_start_idx', 1)
        # index of last layer to have batchnorm
        self.bn_end_idx = self.get('bn_end_idx', -1)

