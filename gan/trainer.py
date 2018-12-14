from gan.losses import GeneratorLoss, DiscriminatorLoss
import torch
from pathlib import Path
import traceback
from timeit import default_timer as timer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class GanTrainer(object):
    def __init__(self, generator, discriminator, config, noise_sampler, lr=0.0002, visualizer=None, estimator=None, seed=1):
        """
        GanTrainer class can be used for training conditional or unconditional gan
        :param generator: generator network, takes z noise as input if unconditional and z, y if conditional
        :param discriminator: discriminator network, takes img as input if unconditional and img, y if conditional
        :param config:
        :param noise_sampler: class with sample_batch(batch_size) that returns (z,) tuple for unconditional gan
               and (z, y) tuple for conditional
        :param lr: learning rate
        :param visualizer: visualizer class that supports update_losses(g_loss, d_loss) and show_generator_results(generator)
        """
        self.device = torch.device(config.DEVICE)
        self.generator = generator
        self.discriminator = discriminator
        #self.generator.apply(weights_init)
        #self.discriminator.apply(weights_init)

        self.config = config
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                                            lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                            lr=lr, betas=(0.5, 0.999))
        self.visualizer = visualizer
        self.estimator = estimator
        self.current_epoch = 0
        self.noise_sampler = noise_sampler
        self.generator_template = 'generator_%d.pth'
        self.discriminator_template = 'discriminator_%d.pth'
        self.checkpoint_template = 'checkpoint_%d.pth'
        self.freezing_thresh = 0.7
        self.seed = seed
        self.scores = []
        self.training_time = 0.0
        self.validation_time = 0.0
        torch.backends.cudnn.deterministic = True

    def train(self, train_loader, valid_loader=None, n_epochs=10, save_interval=5):
        with torch.cuda.device(self.device.index):
            try:
                last_epoch = self.current_epoch
                for self.current_epoch in range(last_epoch + 1, n_epochs + 1):
                    self.train_epoch(train_loader)
                    if valid_loader is not None:
                        self.valid_epoch(valid_loader)
                    if self.current_epoch == n_epochs or self.current_epoch % save_interval == 0:
                        self.save_checkpoint()
            except Exception as e:
                traceback_str = '<br>'.join(traceback.format_tb(e.__traceback__))
                if self.visualizer is not None:
                    self.visualizer.log_text(traceback_str + '<br>' + str(e))
                raise e

    def load_last_checkpoint(self):
        model_path = self.config.MODEL_PATH
        last_epoch = self.get_last_checkpoint(model_path)
        if last_epoch >= 0:
            self.load_checkpoint(last_epoch)

    def train_epoch(self, loader):
        start = timer()
        torch.manual_seed(self.seed + self.current_epoch)
        self.noise_sampler.manual_seed(self.seed + self.current_epoch)
        self.generator.train()
        self.discriminator.train()
        G_train_loss = 0.0
        D_train_loss = 0.0
        n_d_steps = 0
        n_g_steps = 0
        # counter to support k discriminator updates for one generator update
        k_it = 0
        generator_loss = GeneratorLoss()
        discriminator_loss = DiscriminatorLoss(label_smoothing=self.config.label_smoothing)

        for sample in loader:
            sample = (s.cuda() for s in sample)
            # sample is (img,) tuple for regular gan
            # and (img, y) tuple for conditional gan
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()

            D_real, D_logit_real = self.discriminator(*sample)

            D_fake, D_logit_fake = self.discriminator_on_fake(loader.batch_size)

            D_loss = discriminator_loss(D_logit_real, D_logit_fake)
            D_train_loss += D_loss.data
            D_loss.backward()
            self.d_optimizer.step()
            n_d_steps += 1
            k_it += 1

            if k_it == self.config.k:
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                D_fake, D_logit_fake = self.discriminator_on_fake(loader.batch_size)
                G_loss = generator_loss(D_logit_fake)
                G_train_loss += G_loss.data

                G_loss.backward()
                self.g_optimizer.step()
                k_it = 0
                n_g_steps += 1

            # reduce GPU memory usage
            del sample, D_real, D_logit_real, D_loss, D_fake, D_logit_fake

        end = timer()
        self.training_time += end - start
        if self.visualizer is not None:
            self.visualizer.update_losses(epoch=self.current_epoch,
                                          g_loss=G_train_loss / n_g_steps,
                                          d_loss=D_train_loss / n_d_steps,
                                          type='train')
            self.visualizer.show_generator_results(generator=self.generator)
            self.visualizer.update_plot(self.current_epoch, end - start, 'training_time')

    def valid_epoch(self, loader, compute_losses=False):
        start = timer()
        if compute_losses:
            self.generator.eval()
            self.discriminator.eval()
            G_valid_loss = 0.0
            D_valid_loss = 0.0
            n_steps = 0
            generator_loss = GeneratorLoss()
            discriminator_loss = DiscriminatorLoss(label_smoothing=self.config.label_smoothing)
            for sample in loader:
                sample = (s.cuda() for s in sample)
                D_real, D_logit_real = self.discriminator(*sample)
                D_fake, D_logit_fake = self.discriminator_on_fake(loader.batch_size)

                D_loss = discriminator_loss(D_logit_real, D_logit_fake)
                D_valid_loss += D_loss.data.cpu().numpy()

                D_fake, D_logit_fake = self.discriminator_on_fake(loader.batch_size)
                G_loss = generator_loss(D_logit_fake)
                G_valid_loss += G_loss.data.cpu().numpy()
                n_steps += 1
                # reduce GPU memory usage
                del sample, D_real, D_logit_real, D_loss, D_fake, D_logit_fake, G_loss

            G_valid_loss = G_valid_loss / n_steps
            D_valid_loss = D_valid_loss / n_steps
            if self.visualizer is not None:
                self.visualizer.update_losses(epoch=self.current_epoch,
                                              g_loss=G_valid_loss,
                                              d_loss=D_valid_loss,
                                              type='validation')
            else:
                print("%d epoch Validation Losses: G: %d, D: %d" % (self.current_epoch, G_valid_loss, D_valid_loss))

        if self.estimator:
            score = self.estimator.score(self.generator, loader)
            self.scores.append(score)

            if self.visualizer:
                self.visualizer.update_plot(self.current_epoch, score, 'FID')
        end = timer()
        self.validation_time += end - start
        if self.visualizer:
            self.visualizer.update_plot(self.current_epoch, end - start, 'validation_time')

    def discriminator_on_fake(self, batch_size):
        # noise for unconditional gan is (z,) tuple with random noise vector from uniform distribution [-1, 1]
        # for conditional gan noise is (z, y) tuple where y is conditional vector defined by config.conditions
        noise = self.noise_sampler.sample_batch(batch_size)
        noise = [c.cuda() for c in noise]
        # empty tuple for not conditional gan
        condition = noise[1:]
        G_sample = self.generator(*noise)
        D_fake, D_logit_fake = self.discriminator(G_sample, *condition)
        return D_fake, D_logit_fake

    def save_checkpoint(self):
        """
        Save gan checkpoint for continuous training
        """

        save_path = Path(self.config.MODEL_PATH)
        save_path.mkdir(exist_ok=True)
        generator_path = self.generator_template % (self.current_epoch,)
        discriminator_path = self.discriminator_template % (self.current_epoch,)
        torch.save(self.generator.state_dict(), str(save_path / generator_path))
        torch.save(self.discriminator.state_dict(), str(save_path / discriminator_path))
        visdom_env = ""
        if self.visualizer is not None:
            self.visualizer.log_text('Saved checkpoint %d' % (self.current_epoch, ))
            visdom_env = self.visualizer.env_name
            self.visualizer.save()
        state = {
            'epoch': self.current_epoch,
            'generator': str(generator_path),
            'discriminator': str(discriminator_path),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'visdom_env': visdom_env,
            'seed': self.seed,
            'scores': self.scores,
            'training_time': self.training_time,
            'validation_time': self.validation_time
        }
        checkpoint_path = self.checkpoint_template % (self.current_epoch,)
        torch.save(state, str(save_path / checkpoint_path))
        print('Model saved ' + str(save_path / checkpoint_path))

    def load_checkpoint(self, epoch):
        """
        Load gan checkpoint for continuous training
        :param epoch: epoch to load
        """
        load_path = Path(self.config.MODEL_PATH)
        checkpoint_path = self.checkpoint_template % (epoch,)
        state = torch.load(str(load_path / checkpoint_path))
        epoch = state['epoch']
        generator_path = state['generator']
        discriminator_path = state['discriminator']
        self.generator.load_state_dict(torch.load(str(load_path / generator_path)))
        self.discriminator.load_state_dict(torch.load(str(load_path / discriminator_path)))
        self.g_optimizer.load_state_dict(state['g_optimizer'])
        self.d_optimizer.load_state_dict(state['d_optimizer'])
        self.seed = state.get('seed', 1)
        self.scores = state.get('scores', [])
        self.training_time = state.get('training_time', 0.0)
        self.validation_time = state.get('validation_time', 0.0)
        visdom_env = state.get('visdom_env')
        self.current_epoch = epoch
        if self.visualizer is not None and visdom_env and not self.config.NEW_VISDOM_ENV:
            self.visualizer.set_env(visdom_env)

    @staticmethod
    def get_last_checkpoint(path):
        path = Path(path)
        list_files = path.glob('checkpoint_*')
        epochs = [int(str(s).split('_')[-1].split('.')[0]) for s in list_files]
        if not epochs:
            return -1
        return max(epochs)

