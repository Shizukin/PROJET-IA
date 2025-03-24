import pyro.contrib.gp as gp
import torch
import pyro

# Données fictives
X = torch.linspace(-3, 3, 100).reshape(-1, 1)
y = torch.sin(X) + 0.2 * torch.randn(X.shape)

# Modèle GP
kernel = gp.kernels.RBF(input_dim=1, lengthscale=1.0)

# Noise en tant que tensor
noise = torch.tensor(0.1)  # noise doit être un tensor, pas un float
gpr = gp.models.GPRegression(X, y, kernel, noise=noise)

# Inférence bayésienne
optim = pyro.optim.Adam({"lr": 0.01})
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
svi = pyro.infer.SVI(gpr.model, gpr.guide, optim, loss=loss_fn)

# Boucle d'entraînement (exemple)
num_epochs = 1000
for epoch in range(num_epochs):
    loss = svi.step()  # Effectuer une étape de l'inférence
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss}")
