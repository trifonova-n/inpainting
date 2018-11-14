from gan import gan, conditional_gan
from torchsummary import summary


def test_generator_structure():
    z_size = 100
    generator_old = gan.Generator5Net(z_size).cuda()
    generator_new = gan.GeneratorNet(z_size).cuda()

    old_summary = summary(generator_old, input_size=(z_size,))
    new_summary = summary(generator_new, input_size=(z_size,))
    assert old_summary == new_summary


def test_discriminator_structure():
    discriminator_old = gan.Discriminator5().cuda()
    discriminator_new = gan.DiscriminatorNet().cuda()

    old_summary = summary(discriminator_old, input_size=(3, 64, 64))
    new_summary = summary(discriminator_new, input_size=(3, 64, 64))
    assert old_summary == new_summary


def ftest_cond_generator_structure(capsys):
    z_size = 100
    y_size = 5
    generator_old = conditional_gan.Generator5Net(z_size, y_size=y_size).cuda()
    generator_new = conditional_gan.GeneratorNet(z_size, y_size=y_size).cuda()

    summary(generator_old, input_size=(z_size,))
    captured = capsys.readouterr()
    old_summary = captured.out
    summary(generator_new, input_size=(z_size,))
    captured = capsys.readouterr()
    new_summary = captured.out
    print(old_summary)
    print(new_summary)
    assert old_summary == new_summary


def test_cond_discriminator_structure(capsys):
    y_size = 5
    discriminator_uncond = gan.Discriminator5().cuda()
    discriminator_cond = conditional_gan.Discriminator5(y_size).cuda()

    summary(discriminator_uncond, input_size=(3, 64, 64))
    captured = capsys.readouterr()
    old_summary = captured.out
    summary(discriminator_cond, input_size=[(3, 64, 64), (y_size,)])
    captured = capsys.readouterr()
    new_summary = captured.out
    print(old_summary)
    print(new_summary)
    assert old_summary == new_summary