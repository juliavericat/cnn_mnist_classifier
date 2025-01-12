import torch
import typer
from data import corrupt_mnist
from cnn_mnist.model_v1 import SimpleCNN
from torch import nn, optim
import matplotlib.pyplot as plt


app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, epochs: int = 10, batch_size: int = 32) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracies, train_steps = [], [], []
    step = 0
    for epoch in range(epochs):
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
                    f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {acc_train*100:.2f}%"
                )

    print("Training complete")
    torch.save(model.state_dict(), "models/trained_model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(train_accuracies)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    
if __name__ == "__main__":
    app()

