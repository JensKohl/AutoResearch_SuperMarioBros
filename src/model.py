import torch.nn as nn


class PolicyModel(nn.Module):
    """Dueling DQN: shared conv+FC backbone with separate value and advantage streams.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a)). Reference: Wang et al. 2016
    https://arxiv.org/abs/1511.06581
    """
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
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

    def forward(self, x):
        feat = self.fc(self.conv(x).view(x.size(0), -1))
        v = self.value(feat)
        a = self.advantage(feat)
        return v + a - a.mean(dim=1, keepdim=True)
