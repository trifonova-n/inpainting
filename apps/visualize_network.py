from inpainting.visualizer import Visualizer
from argparse import ArgumentParser
from inpainting.dataset import ConditionSampler, Data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', default='conditional_model')
    parser.add_argument('--epoch', default='last')
    args = parser.parse_args()
    DATA_PATH = 'data/img_align_celeba'
    Z_SIZE = 100

    data = Data(DATA_PATH, Z_SIZE)
    y_sampler = ConditionSampler(data=data)
    visualizer = Visualizer('visualize_network', y_sampler=y_sampler)
    visualizer.update_losses(2., 7., type='validation')
    visualizer.update_losses(4., 8., type='validation')
    visualizer.update_losses(6., 9., type='validation')
    visualizer.save()
    s = input('--> ')

