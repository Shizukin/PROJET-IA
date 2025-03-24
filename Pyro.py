import torch
import pyro 
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn.functional as F
class BayesianNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.fc1 = PyroModule[torch.nn.Linear](1, 128)
        self.fc2 = PyroModule[torch.nn.Linear](128, 1)
        
        # Définition des priors
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([128, 1]).to_event(1))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([128]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 128]).to_event(1))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x_data):
        x = F.relu(self.fc1(x_data))
        return self.fc2(x)

# Données d'exemple
x_data = torch.randn(100, 1)
y_data = 3 * x_data + torch.randn(100, 1) * 0.1

# Modèle bayésien avec pyro
def model(x_data, y_data):
    # Les priors sont déjà définis dans le modèle
    output = BayesianNN()(x_data)
    pyro.sample("obs", dist.Normal(output, 0.1), obs=y_data)

# Inference
from pyro.infer import MCMC, NUTS

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=50)
mcmc.run(x_data, y_data)

# Affichage des résultats
posterior_samples = mcmc.get_samples()
print(posterior_samples)
