from inpainting.visualizer import Visualizer
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', default='conditional_model')
    parser.add_argument('--epoch', default='last')
    args = parser.parse_args()

    visualizer = Visualizer('visualize_network')
    visualizer.update_losses(2., 7., type='validation')
    visualizer.update_losses(4., 8., type='validation')
    visualizer.update_losses(6., 9., type='validation')
    s = input('--> ')

