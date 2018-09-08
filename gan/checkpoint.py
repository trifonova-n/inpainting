import torch
from pathlib import Path

generator_template = 'generator_%d.pth'
discriminator_template = 'discriminator_%d.pth'
checkpoint_template = 'checkpoint_%d.pth'


def save_checkpoint(trainer, save_path):
    """
    Save gan checkpoint for continuous training
    :param trainer: object of type derived from BaseGanTrainer
    :param save_path: directory to save checkpoint
    """

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    generator_path = generator_template % (trainer.current_epoch,)
    discriminator_path = discriminator_template % (trainer.current_epoch,)
    torch.save(trainer.generator.state_dict(), str(save_path / generator_path))
    torch.save(trainer.discriminator.state_dict(), str(save_path / discriminator_path))
    visdom_env = ""
    if trainer.visualizer is not None:
        visdom_env = trainer.visualizer.env_name
        trainer.visualizer.save()
    state = {
        'epoch': trainer.current_epoch,
        'generator': str(generator_path),
        'discriminator': str(discriminator_path),
        'g_optimizer': trainer.g_optimizer.state_dict(),
        'd_optimizer': trainer.d_optimizer.state_dict(),
        'visdom_env': visdom_env
    }
    checkpoint_path = checkpoint_template % (trainer.current_epoch,)
    torch.save(state, str(save_path / checkpoint_path))


def load_checkpoint(load_path, epoch, trainer):
    """
    Load gan checkpoint for continuous training
    :param load_path: directory containing checkpoints
    :param epoch: epoch to load
    :param trainer: object of type derived from BaseGanTrainer
    """
    load_path = Path(load_path)
    checkpoint_path = checkpoint_template % (epoch,)
    state = torch.load(str(load_path / checkpoint_path))
    epoch = state['epoch']
    generator_path = state['generator']
    discriminator_path = state['discriminator']
    trainer.generator.load_state_dict(torch.load(str(load_path / generator_path)))
    trainer.discriminator.load_state_dict(torch.load(str(load_path / discriminator_path)))
    trainer.g_optimizer.load_state_dict(state['g_optimizer'])
    trainer.d_optimizer.load_state_dict(state['d_optimizer'])
    visdom_env = state.get('visdom_env')
    trainer.current_epoch = epoch
    if trainer.visualizer is not None and visdom_env:
        trainer.visualizer.set_env(visdom_env)


def get_last_checkpoint(path):
    path = Path(path)
    list_files = path.glob('checkpoint_*')
    epochs = [int(str(s).split('_')[-1].split('.')[0]) for s in list_files]
    if not epochs:
        return -1
    return max(epochs)
