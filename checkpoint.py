import torch
from pathlib import Path

generator_template = 'generator_%d.pth'
discriminator_template = 'discriminator_%d.pth'
checkpoint_template = 'checkpoint_%d.pth'


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, save_path):
    """
    Save gan checkpoint for continuous training
    :param generator: model
    :param discriminator: model
    :param g_optimizer: Optimizer for generator
    :param d_optimizer: Optimizer for discriminator
    :param epoch: current epoch
    :param save_path: directory to save checkpoint
    """
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    generator_path = generator_template % (epoch,)
    discriminator_path = discriminator_template % (epoch,)
    torch.save(generator.state_dict(), str(save_path / generator_path))
    torch.save(discriminator.state_dict(), str(save_path / discriminator_path))
    state = {
        'epoch': epoch,
        'generator': str(generator_path),
        'disriminator': str(discriminator_path),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
    }
    checkpoint_path = checkpoint_template % (epoch,)
    torch.save(state, str(save_path / checkpoint_path))


def load_checkpoint(load_path, epoch, generator, discriminator, g_optimizer, d_optimizer):
    """
    Load gan checkpoint for continuous training
    :param load_path: directory containing checkpoints
    :param epoch: epoch to load
    :param generator: model
    :param discriminator: model
    :param g_optimizer: Optimizer for generator
    :param d_optimizer: Optimizer for discriminator
    """
    load_path = Path(load_path)
    checkpoint_path = checkpoint_template % (epoch,)
    state = torch.load(str(load_path / checkpoint_path))
    epoch = state['epoch']
    generator_path = Path(state['generator'])
    discriminator_path = Path(state['discriminator'])
    g_optimizer.load_state_dict(state['g_optimizer'])
    d_optimizer.load_state_dict(state['d_optimizer'])
    generator.load_state_dict(str(load_path / generator_path))
    discriminator.load_state_dict(str(load_path / discriminator_path))


def get_last_checkpoint(path):
    path = Path(path)
    list_files = path.glob('checkpoint_*')
    epochs = [int(str(s).split('_')[1].split('.')[0]) for s in list_files]
    if not epochs:
        return -1
    return max(epochs)
