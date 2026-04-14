import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    """Dueling DQN Q-network: shared CNN + separate value and advantage streams.

    Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    Dueling architecture (Wang et al. 2016) gives more accurate Q-values by
    separately estimating state value V(s) and per-action advantage A(s,a).
    """
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        f = self.conv(x).view(x.size(0), -1)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(1, keepdim=True)
