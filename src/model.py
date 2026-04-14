import torch.nn as nn


class PolicyModel(nn.Module):
    """Very simple CNN: 2 conv layers + 1 FC layer."""
    def __init__(self, n_actions):
        super(PolicyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features.view(features.size(0), -1))
