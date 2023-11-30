import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from model import GavelModel


def load_data(valid_data_path: str) -> TensorDataset:
    """_summary_

    Args:
        train_data_path (str): train data for npz format
        test_data_path (str): test data for npz format

    Returns:
        TensorDataset: for GPU training, transform to tensor
    """

    valid_data = np.load(valid_data_path)

    x_data = valid_data["x_valid"]
    y_data = valid_data["y_valid"]

    # transform into pytorch tensor data
    x_data = torch.tensor(x_data).permute(0, 3, 1, 2).float()
    y_data = torch.tensor(y_data).long()

    # make TensorDataset
    valid_dataset = TensorDataset(x_data, y_data)

    return valid_dataset


def validate(
    model,
    device,
    valid_loader,
) -> None:
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    # start validating
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test data: {accuracy:.2f}%")


if __name__ == "__main__":
    # convert data info DataLoader
    valid_data_path = "test_dataset1_cow157.npz"

    valid_dataset = load_data(valid_data_path)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # build model and set config hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GavelModel().to(device)

    model_path = "model-ep150-val_loss0.265-val_acc0.896.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # start training
    validate(
        model=model,
        device=device,
        valid_loader=valid_loader,
    )
