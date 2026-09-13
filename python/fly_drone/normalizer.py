import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NeuralNormalizer(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        n = observation_space.shape[0]
        super().__init__(observation_space, n)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))

    def forward(self, observations):
        return (observations - self.mean) / self.scale
