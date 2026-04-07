import pytest
from mlx import nn


class _DummyTransformer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.layer = linear


@pytest.fixture
def DummyTransformer():
    return _DummyTransformer
