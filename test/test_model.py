import pytest
from gan import gan, conditional_gan
from torchsummary import summary
import re


@pytest.fixture(params=[(1024, 4, 4), (1024, 2, 2), (512, 4, 4), (512, 2, 2)])
def min_shape(request):
    return request.param


@pytest.fixture(params=[2, 1.5])
def channel_scaling_factor(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def bn_start_idx(request):
    return request.param


@pytest.fixture(params=[-1, -2, -3])
def bn_end_idx(request):
    return request.param


@pytest.fixture(params=[32, 64])
def start_channels(request):
    return request.param


@pytest.fixture(params=[2, 4])
def feature_img_size(request):
    return request.param


@pytest.fixture(params=[100, 150])
def z_size(request):
    return request.param


@pytest.fixture
def generator_hyperparams(min_shape, channel_scaling_factor, bn_start_idx, bn_end_idx, z_size):
    return {'min_shape': min_shape,
            'channel_scaling_factor': channel_scaling_factor,
            'bn_start_idx': bn_start_idx,
            'bn_end_idx': bn_end_idx,
            'z_size': z_size}


@pytest.fixture
def discriminator_hyperparams(start_channels, channel_scaling_factor, feature_img_size, bn_start_idx, bn_end_idx):
    return {'start_channels': start_channels,
            'channel_scaling_factor': channel_scaling_factor,
            'feature_img_size': feature_img_size,
            'bn_start_idx': bn_start_idx,
            'bn_end_idx': bn_end_idx}


def test_generator(generator_hyperparams):
    z_size = generator_hyperparams['z_size']
    generator = gan.GeneratorNet(params=generator_hyperparams).cuda()
    info = list(summary(generator, input_size=(z_size,)).items())
    first_layer = info[0][1]
    last_layer = info[-1][1]
    assert first_layer['input_shape'] == [-1, z_size]
    assert last_layer['output_shape'] == [-1, 3, 64, 64]
    idx = 1
    i = 0
    if generator.hyper_params.bn_start_idx == 0:
        i += 1
        assert re.match('^BatchNorm1d-\d+', info[i][0])
    i += 1
    assert re.match('^ReLU-\d+', info[i][0])
    i += 1
    expected_input_shape = [-1, *generator_hyperparams['min_shape']]
    while expected_input_shape[2] < 64:
        assert re.match('^ConvTranspose2d-\d+', info[i][0])
        input_shape = info[i][1]['input_shape']
        assert input_shape == expected_input_shape
        i += 1
        if generator.hyper_params.bn_start_idx <= idx <= generator.hyper_params.bn_end_idx:
            assert re.match('^BatchNorm2d-\d+', info[i][0])
            i += 1
        assert re.match('^ReLU-\d+', info[i][0])
        output_shape = info[i][1]['output_shape']
        expected_output_shape = [expected_input_shape[0],
                                 int(expected_input_shape[1]/generator_hyperparams['channel_scaling_factor']),
                                 expected_input_shape[2]*2,
                                 expected_input_shape[3]*2]
        assert output_shape == expected_output_shape
        expected_input_shape = expected_output_shape
        i += 1
        idx += 1


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