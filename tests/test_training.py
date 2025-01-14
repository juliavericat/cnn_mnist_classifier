import os
import torch
import pytest
from cnn_mnist.model_v1 import SimpleCNN
from cnn_mnist.train import train  
from cnn_mnist.data import corrupt_mnist


@pytest.fixture
def mock_training_data():
    """Create a small dataset for testing."""
    train_set, _ = corrupt_mnist()
    return torch.utils.data.DataLoader(train_set, batch_size=2)


def test_model_initialization():
    """Test that the model initializes correctly."""
    model = SimpleCNN()
    assert isinstance(model, torch.nn.Module), "Model is not a PyTorch module."
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"


def test_training_loop(mock_training_data):
    """Test a single epoch of the training loop."""
    model = SimpleCNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    try:
        for img, target in mock_training_data:
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
        assert True, "Training loop ran successfully."
    except Exception as e:
        pytest.fail(f"Training loop failed: {e}")


def test_model_saving(tmp_path):
    """Test that the model is saved correctly."""
    model = SimpleCNN()
    model_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), model_path)
    assert os.path.exists(model_path), "Model file was not saved."

