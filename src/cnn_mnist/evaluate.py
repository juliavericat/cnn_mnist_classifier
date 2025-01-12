import torch
import typer
from data import corrupt_mnist
from cnn_mnist.model_v1 import SimpleCNN
from torch import nn


app = typer.Typer()


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == labels).float().sum().item()
            total_samples += labels.size(0)

        # Average loss and accuracy
        avg_loss = total_loss / len(test_dataloader)
        accuracy = (total_correct / total_samples) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Validation loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    app()
