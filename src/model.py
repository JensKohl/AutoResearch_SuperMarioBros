import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    """Dueling DQN architecture.

    Splits into a value stream V(s) and advantage stream A(s,a),
    combined as Q(s,a) = V(s) + A(s,a) - mean(A).
    This helps the agent learn which states are valuable independent
    of the specific action taken. Reference: Wang et al. 2016
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, n_actions):
        super(PolicyModel, self).__init__()
        # Input: 4 stacked grayscale frames (4x84x84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        v = self.value(features)
        a = self.advantage(features)
        return v + a - a.mean(dim=1, keepdim=True)
