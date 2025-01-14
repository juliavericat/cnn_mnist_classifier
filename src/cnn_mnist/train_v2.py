import wandb
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
    # Initialize WandB
    wandb.login()
    wandb.init(
        project="cnn-mnist-classifier",  
        entity="juliavericatg-danmarks-tekniske-universitet-dtu", 
        config={
            "lr": cfg.train.lr,
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size
        },
        job_type='Train'
    )

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
    epoch_losses, epoch_accuracies, train_steps = [], [], []
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
            epoch_losses.append(loss.item())
            acc_train = (output.argmax(dim=1) == labels).float().mean().item()
            epoch_accuracies.append(acc_train)
            step += 1

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {loss.item():.4f}, Accuracy: {acc_train*100:.2f}%")
                # Log metrics to WandB
                wandb.log({"accuracy": acc_train, "loss": loss.item(), "epoch": epoch + 1})

    print("Training complete")
    model_path = "models/trained_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Log the trained model to WandB
    artifact = wandb.Artifact(name="trained_model", type="model")
    artifact.add_file(model_path)
    artifact.save()

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    axs[0].plot(epoch_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(epoch_accuracies)
    axs[1].set_title("Train accuracy")
    plt.savefig("reports/figures/training_results.png")

    fig_path = "reports/figures/training_results.png"
    wandb.log({"Training Results": wandb.Image(fig_path)})

    # Finish the WandB run
    wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def train_model_with_hydra(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    train_model(cfg)

@app.command()
def train(
    batch_size: int = typer.Option(..., help="Batch size for training"),
    epochs: int = typer.Option(..., help="Number of epochs for training"),
    learning_rate: float = typer.Option(..., help="Learning rate for training")
) -> None:
    """Train a model on MNIST using Hydra configuration."""
    overrides = [
        f"train.batch_size={batch_size}",
        f"train.epochs={epochs}",
        f"train.learning_rate={learning_rate}"
    ]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="conf")
    cfg = hydra.compose(config_name="train_conf", overrides=overrides)
    train_model_with_hydra(cfg)

if __name__ == "__main__":
    app()