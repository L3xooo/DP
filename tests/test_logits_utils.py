import pytest
import torch
from td3.utils.logits_utils import logits_to_weights, add_logit_noise


@pytest.mark.parametrize("length", [10, 50, 100, 500, 1000])
def test_logits_to_weights_sums_to_one(length):
    logits = torch.randn(length)
    weights = logits_to_weights(logits)

    assert weights.shape == logits.shape
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


@pytest.mark.parametrize("noise_std", [0, 0.0, -1.0, -0.5, None])
def test_no_noise_added_when_std_invalid(noise_std):
    logits = torch.randn(100)
    result = add_logit_noise(logits, noise_std=noise_std, noise_clip=1.0)

    assert torch.equal(result, logits)


@pytest.mark.parametrize(
    "noise_std, noise_clip",
    [
        (0.1, 0.1),
        (0.5, 0.5),
        (1.0, 1.0),
        (5.0, 1.0),
        (10.0, 0.01),
        (0.01, 5.0),
    ],
)
def test_noise_added_when_std_positive(noise_std, noise_clip):
    torch.manual_seed(42)
    logits = torch.randn(1000)
    result = add_logit_noise(logits, noise_std=noise_std, noise_clip=noise_clip)

    assert not torch.equal(result, logits)


@pytest.mark.parametrize(
    "noise_std, noise_clip",
    [
        (0.1, 0.1),
        (0.5, 0.5),
        (1.0, 1.0),
        (5.0, 1.0),
        (10.0, 0.01),
        (0.01, 5.0),
    ],
)
def test_noise_never_exceeds_clip(noise_std, noise_clip):
    torch.manual_seed(42)
    logits = torch.randn(1000)

    result = add_logit_noise(logits, noise_std=noise_std, noise_clip=noise_clip)
    applied_noise = (result - logits).abs()

    assert applied_noise.max().item() <= noise_clip + 1e-6
