from src.privacy import GaussianMechanism


def test_calibrate_sigma_is_positive():
    sigma = GaussianMechanism.calibrate_sigma(1.0, 0.1, 1e-6)
    assert sigma > 0


def test_zero_sigma_returns_same_tensor():
    import torch

    x = torch.ones(4)
    y = GaussianMechanism.add_noise(x, 0.0)
    assert torch.equal(x, y)
