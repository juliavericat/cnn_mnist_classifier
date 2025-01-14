import matplotlib.pyplot as plt
import torch
import typer
import wandb
from cnn_mnist.data import corrupt_mnist
from cnn_mnist.model_v1 import SimpleCNN
from sklearn.metrics import RocCurveDisplay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(learning_rate: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{learning_rate=}, {batch_size=}, {epochs=}")
    wandb.init(
        project="cnn_mnist_classifier",
        config={"lr": learning_rate, "batch_size": batch_size, "epochs": epochs},
    )

    model = SimpleCNN().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                # grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                # wandb.log({"gradients": wandb.Histogram(grads)})
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0).cpu()
                wandb.log({"gradients": wandb.Histogram(grads)})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0).cpu()
        targets = torch.cat(targets, 0).cpu()

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot.numpy(),
                preds[:, class_id].numpy(),
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )

        # alternatively use wandb.log({"roc": wandb.Image(plt)}
        wandb.log({"roc": wandb.Image(plt)})
        plt.close()  # close the plot to avoid memory leaks and overlapping figures

    print("Training complete")
    model_path = "models/trained_model.pth"
    torch.save(model.state_dict(), model_path)

    # Log the trained model to WandB
    artifact = wandb.Artifact(name="new_trained_model", type="model")
    artifact.add_file(model_path)
    artifact.save()

if __name__ == "__main__":
    typer.run(train)