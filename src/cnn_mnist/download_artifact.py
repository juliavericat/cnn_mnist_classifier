import wandb
import torch
from model_v1 import SimpleCNN

run = wandb.init()
artifact = run.use_artifact('juliavericatg-danmarks-tekniske-universitet-dtu/cnn-mnist-classifier/new_trained_model:v1', type='model')
artifact_dir = artifact.download("artifact_dir")
model = SimpleCNN()
model.load_state_dict(torch.load("artifact_dir/trained_model.pth"))