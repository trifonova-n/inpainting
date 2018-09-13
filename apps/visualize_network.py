from inpainting.visualizer import Visualizer
from argparse import ArgumentParser
from inpainting.dataset import ConditionSampler, Data
import inpainting.celeba_config as conf
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', default='conditional_model')
    parser.add_argument('--epoch', default='last')
    args = parser.parse_args()
    DATA_PATH = 'data/img_align_celeba'
    Z_SIZE = 100

    data = Data(DATA_PATH, Z_SIZE)
    loader = DataLoader(data, batch_size=4, num_workers=conf.NUM_WORKERS, shuffle=True)
    y_sampler = ConditionSampler(data=data, z_size=Z_SIZE)

    conf.ENV_NAME = 'visualize_network'
    visualizer = Visualizer(conf, noise_sampler=y_sampler)
    visualizer.set_env('visualize_network')
    visualizer.log_text("1, 2, 3")
    visualizer.log_text("4,<br> 5, 6")
    visualizer.update_losses(2., 7., type='validation')
    visualizer.update_losses(4., 8., type='validation')
    visualizer.update_losses(6., 9., type='validation')
    batch, = next(iter(loader))
    visualizer.plot_batch(batch, ['img1', 'img2', 'img3', 'img4'])
    print(visualizer.env_name)
    print('data:', visualizer.vis.win_hash(win=visualizer.valid_losses_plt, env=visualizer.env_name))
    visualizer.save()
    s = input('--> ')

