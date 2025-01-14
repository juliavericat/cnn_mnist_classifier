import torch
import pytest
import os.path
from cnn_mnist.data import corrupt_mnist

file_path ="/home/jvg/Desktop/MLOps/W1/D2_Organisation_VersionControl/cnn_mnist_classifier/data"

@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    # assert len(dataset) == N_train for training and N_test for test
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    
    # assert that each datapoint has shape [1,28,28] or [784] 
    # depending on how you choose to format
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Incorrect shape of the datapoints in train set"
            assert y in range(10), "Incorrect shape of the datapoints in test set"
    
    # assert that all labels are represented
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "There are missing labels in the train set"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "There are missing labels in the test set"