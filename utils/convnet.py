import gym
import torch
from torch import nn
from gym import spaces

from torchvision.ops import DropBlock2d

class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        img_size: int,
        features_dim: int = 512,
        with_norm_layer=False,
        p_dropblock=0.0
    ) -> None:
        super().__init__()
        assert features_dim > 0
        self._img_size = img_size
        self._features_dim = features_dim
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=6, stride=3, padding=0),
            DropBlock2d(p_dropblock, 3),
            nn.BatchNorm2d(32) if with_norm_layer else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            DropBlock2d(p_dropblock, 3),
            nn.BatchNorm2d(64) if with_norm_layer else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            DropBlock2d(p_dropblock, 3),
            nn.BatchNorm2d(64) if with_norm_layer else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.ones((1, n_input_channels, self._img_size, self._img_size)).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
