import torch
import pytest
from cnn_mnist.model_v1 import SimpleCNN

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = SimpleCNN()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)

def test_error_on_wrong_shape():
    model = SimpleCNN()    
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape \[1, 28, 28\]'):
        model(torch.randn(1,1,28,29))