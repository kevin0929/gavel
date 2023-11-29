import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from model import GavelModel

# filter pandas warning msg
warnings.filterwarnings("ignore")


def load_data(train_data_path: str, test_data_path: str) -> TensorDataset:
    """_summary_

    Args:
        train_data_path (str): train data for npz format
        test_data_path (str): test data for npz format

    Returns:
        TensorDataset: for GPU training, transform to tensor
    """

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    x_train = train_data["x_train"]
    y_train = train_data["y_train"]
    x_test = test_data["x_test"]
    y_test = test_data["y_test"]

    # transform into pytorch tensor data
    x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float()
    y_train = torch.tensor(y_train).long()
    x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float()
    y_test = torch.tensor(y_test).long()

    # make TensorDataset
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, test_dataset


def plot(train_loss: list, test_accuracies: list) -> None:
    """
    plot training detail (loss and test accuracy)
    """

    plt.figure(figsize=(12, 6))

    # Plot loss detail graph
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot test accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy", color="orange")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_detail.png")


def train(
    model,
    criterion,
    optimizer,
    epochs,
    train_loader,
    test_loader,
) -> None:
    train_loss = []
    test_accuracies = []

    # start training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

        average_loss = total_loss / len(train_loader)
        train_loss.append(average_loss)

        # test current model
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}"
        )

    # save model
    filepath = (
        f"model-ep{epoch+1:03d}-val_loss{test_loss:.3f}-val_acc{test_accuracy:.3f}.pth"
    )
    torch.save(model.state_dict(), filepath)

    # plot training detain graph
    plot(train_loss, test_accuracies)


if __name__ == "__main__":
    # convert data into DataLoader
    train_data_path = "train_data.npz"
    test_data_path = "test_data.npz"

    train_dataset, test_dataset = load_data(train_data_path, test_data_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # build model and set config hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GavelModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    epochs = 150

    # start training
    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
    )
