import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

# Définition du réseau de neurones bayésien
class BayesianNN(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, 10)
        # Définir les priors directement sur les poids et biais
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand(self.fc1.weight.shape).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand(self.fc1.bias.shape).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](10, out_features)
        # Définir les priors pour fc2
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand(self.fc2.weight.shape).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand(self.fc2.bias.shape).to_event(1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test du modèle
model = BayesianNN(5, 1)
x = torch.randn(3, 5)  # Trois entrées aléatoires
output = model(x)
print(output)
