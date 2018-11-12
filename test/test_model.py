from gan.gan import Generator5Net, GeneratorNet
from torchsummary import summary


def test_model_structure(capsys):
    z_size = 100
    generator_old = Generator5Net(z_size).cuda()
    generator_new = GeneratorNet(z_size).cuda()

    summary(generator_old, input_size=(z_size,))
    captured = capsys.readouterr()
    old_summary = captured.out
    summary(generator_new, input_size=(z_size,))
    captured = capsys.readouterr()
    new_summary = captured.out
    print(old_summary)
    print(new_summary)
    assert old_summary == new_summary
