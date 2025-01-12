import torch
from torch import nn
import hydra
from omegaconf import DictConfig

class SimpleCNN(nn.Module):
    """My awesome model."""

    def __init__(self, cfg) -> None:
        super().__init__()
        
        # Extracting parameters from the config file
        self.conv1 = nn.Conv2d(
            in_channels=cfg.conv1.in_channels,
            out_channels=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=cfg.conv2.in_channels,
            out_channels=cfg.conv2.out_channels,
            kernel_size=cfg.conv2.kernel_size,
            stride=cfg.conv2.stride
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=cfg.conv3.in_channels,
            out_channels=cfg.conv3.out_channels,
            kernel_size=cfg.conv3.kernel_size,
            stride=cfg.conv3.stride
        )
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc1 = nn.Linear(cfg.fc1.in_features, cfg.fc1.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

@hydra.main(version_base=None, config_path="conf", config_name="model_conf")  
def main(cfg: DictConfig) -> None:
    model = SimpleCNN(cfg.model)
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
