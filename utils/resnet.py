import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, embed_dim, with_norm_layer=False):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(embed_dim) if with_norm_layer else nn.Identity()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(embed_dim) if with_norm_layer else nn.Identity()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return x + out

class ResNetImpala(nn.Module):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    see https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L28
    """

    def __init__(self, img_size, embed_size=256, depths=[16, 32, 32], with_norm_layer = False):
        super(ResNetImpala, self).__init__()
        self.img_size = img_size
        self.conv_layers = self._make_layer(depths, with_norm_layer)
        self.relu = nn.ReLU()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flat_size = torch.flatten(self.conv_layers(torch.ones((1, 3, img_size, img_size)))).shape[0]

        self.linear = nn.Linear(flat_size, embed_size)
    
    def conv_sequence(self, input_size, depth, with_norm_layer):
        self.conv1 = nn.Conv2d(input_size, depth, kernel_size=3, stride=1, padding='same')
        self.bn = nn.BatchNorm2d(depth) if with_norm_layer else nn.Identity()
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res_block1 = ResBlock(depth)
        self.res_block2 = ResBlock(depth)
        return nn.Sequential(self.conv1, self.bn, self.max_pooling, self.res_block1, self.res_block2)

    def _make_layer(self, depths, with_norm_layer):
        layers = []
        input_sizes = [3] + depths[:-1]
        for i, depth in enumerate(depths):
            layers.append(self.conv_sequence(input_sizes[i], depth, with_norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.relu(out)
        out = self.linear(out)
        out = self.relu(out)
        return out