from gan.losses import GeneratorLoss, DiscriminatorLoss
import numpy as np


def test_training(trainer, dataloaders, conf):
    train_loader, valid_loader = dataloaders

    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss(label_smoothing=conf.label_smoothing)
    for sample in train_loader:
        sample = (s.cuda() for s in sample)
        # sample is (img,) tuple for regular gan
        # and (img, y) tuple for conditional gan
        trainer.d_optimizer.zero_grad()
        trainer.g_optimizer.zero_grad()

        D_real, D_logit_real = trainer.discriminator(*sample)

        D_fake, D_logit_fake = trainer.discriminator_on_fake(train_loader.batch_size)

        D_loss = discriminator_loss(D_logit_real, D_logit_fake)
        print(D_loss.data)
        assert np.all(np.isfinite(D_loss.data.cpu().numpy()))
        D_loss.backward()
        for p in trainer.generator.parameters():
            #print(p.grad.data)
            assert np.all(np.isfinite(p.grad.data.cpu().numpy()))

        for p in trainer.discriminator.parameters():
            #print(p.grad.data)
            assert np.all(np.isfinite(p.grad.data.cpu().numpy()))

        trainer.d_optimizer.step()
        for p in trainer.generator.parameters():
            #print(p.data)
            assert np.all(np.isfinite(p.data.cpu().numpy()))

        for p in trainer.discriminator.parameters():
            #print(p.data)
            assert np.all(np.isfinite(p.data.cpu().numpy()))

        trainer.d_optimizer.zero_grad()
        trainer.g_optimizer.zero_grad()
        D_fake, D_logit_fake = trainer.discriminator_on_fake(train_loader.batch_size)
        G_loss = generator_loss(D_logit_fake)
        print(G_loss.data)
        assert np.all(np.isfinite(G_loss.data.cpu().numpy()))
        G_loss.backward()
        trainer.g_optimizer.step()
        for p in trainer.generator.parameters():
            #print(p.data)
            assert np.all(np.isfinite(p.data.cpu().numpy()))

        for p in trainer.discriminator.parameters():
            #print(p.data)
            assert np.all(np.isfinite(p.data.cpu().numpy()))
