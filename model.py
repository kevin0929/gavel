import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=kernels[0],
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels[0],
            out_channels=out_channels[1],
            kernel_size=kernels[1],
            padding="same",
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            kernel_size=kernels[2],
            padding="same",
        )

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x + residual


class GavelModel(nn.Module):
    def __init__(self):
        super(GavelModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=(3, 3))
        self.attention1 = SpatialAttentionBlock(
            in_channels=32, out_channels=[1, 16, 32], kernels=[1, 3, 3]
        )
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.attention2 = SpatialAttentionBlock(
            in_channels=64, out_channels=[1, 32, 64], kernels=[1, 3, 3]
        )
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.attention3 = SpatialAttentionBlock(
            in_channels=64, out_channels=[1, 32, 64], kernels=[1, 3, 3]
        )
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(2, 2))
        self.attention4 = SpatialAttentionBlock(
            in_channels=128, out_channels=[1, 64, 128], kernels=[1, 3, 3]
        )
        self.dropout4 = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.dropout5 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.attention2(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.attention3(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.attention4(x)
        x = self.pool(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)

        return x
