import torch
import hydra
from omegaconf import DictConfig
import typer
from data import corrupt_mnist
from model_v2 import SimpleCNN
from torch import nn, optim
import matplotlib.pyplot as plt

app = typer.Typer()

# Train function with configuration passed as an argument
def train_model(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning rate: {cfg.train.lr}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model using the configuration
    model = SimpleCNN(cfg.model).to(device)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True
    )

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # Track losses and accuracies
    train_losses, train_accuracies, train_steps = [], [], []
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            train_steps.append(step)
            train_losses.append(loss.item())
            acc_train = (output.argmax(dim=1) == labels).float().mean().item()
            train_accuracies.append(acc_train)
            step += 1

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {loss.item():.4f}, Accuracy: {acc_train*100:.2f}%"
                )

    print("Training complete")
    torch.save(model.state_dict(), "models/trained_model.pth")
    
    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(train_accuracies)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def train_model_with_hydra(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    train_model(cfg)

@app.command()
def train() -> None:
    """Train a model on MNIST using Hydra configuration."""
    train_model_with_hydra()

if __name__ == "__main__":
    app()
