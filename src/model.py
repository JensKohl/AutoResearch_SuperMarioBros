import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    """PPO Actor-Critic network: shared CNN + separate policy (actor) and value (critic) heads.

    Architecture matches the previous Dueling DQN CNN so we can warm-start the convolutional
    layers from the existing x=1959 model checkpoint. The FC heads are re-initialized.

    evaluate.py compatibility: forward() returns action logits; argmax == greedy action,
    exactly like a DQN's Q-values.
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
        self.policy = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        # Small init for policy output so initial action distribution is near-uniform
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy[-1].bias)

        # Residual head for beyond-barrier corrections (exp140).
        # Starts at zero — initially a no-op. BC trains this for x>=900 states while
        # policy[-1] stays frozen, giving true isolation between early and late game.
        self.beyond_head = nn.Linear(512, n_actions)
        nn.init.zeros_(self.beyond_head.weight)
        nn.init.zeros_(self.beyond_head.bias)

    def forward(self, x):
        """Returns action logits — evaluate.py uses .max(1)[1] for greedy action."""
        f = self.conv(x).view(x.size(0), -1)
        features = self.policy[:2](f)   # 512D: Linear(3136→512) + ReLU
        return self.policy[-1](features) + self.beyond_head(features)

    def full_forward(self, x):
        """Returns (logits, value) used during PPO training.
        Includes beyond_head so PPO trains the residual head correctly."""
        f = self.conv(x).view(x.size(0), -1)
        features = self.policy[:2](f)
        logits = self.policy[-1](features) + self.beyond_head(features)
        return logits, self.value_head(f)
