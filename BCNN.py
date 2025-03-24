import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.optim as optim

# Modèle de CNN bayésien
class BayesianCNN(PyroModule):
    def __init__(self):
        super().__init__()
        
        # Convolution 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Définir les poids et les biais comme des paramètres PyTorch normaux (optimisables)
        self.conv1.weight = nn.Parameter(torch.randn(self.conv1.weight.shape))
        self.conv1.bias = nn.Parameter(torch.randn(self.conv1.bias.shape))
        
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * 28 * 28, 10)
        # Définir les poids et les biais comme des paramètres PyTorch normaux (optimisables)
        self.fc1.weight = nn.Parameter(torch.randn(self.fc1.weight.shape))
        self.fc1.bias = nn.Parameter(torch.randn(self.fc1.bias.shape))

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Activation de la première couche convolutive
        x = x.view(x.size(0), -1)  # Aplatir pour la couche fully connected
        x = self.fc1(x)  # Passer dans la couche fully connected
        return x

# Création du modèle
model = BayesianCNN()

# Vérifier que les paramètres sont bien accessibles
for name, param in model.named_parameters():
    print(name, param.shape)

# Manuellement ajouter les paramètres optimisables dans l'optimiseur
params = list(model.parameters())  # Récupérer tous les paramètres du modèle

# Vérifier que les paramètres sont bien récupérés
if not params:
    print("Aucun paramètre trouvé dans le modèle.")
else:
    print(f"Nombre de paramètres : {len(params)}")

# Définir un optimiseur (Adam)
optimizer = optim.Adam(params, lr=1e-3)

# Définir une fonction de perte (CrossEntropyLoss pour classification)
criterion = torch.nn.CrossEntropyLoss()

# Simuler un jeu de données d'entraînement (par exemple, batch de 16 images 28x28 avec 1 canal)
# Remplace cela par tes données réelles
x_train = torch.randn(16, 1, 28, 28)  # Batch de 16 images de taille 28x28 en niveaux de gris
y_train = torch.randint(0, 10, (16,))  # Labels aléatoires pour 10 classes

# Boucle d'entraînement
num_epochs = 10  # Nombre d'époques d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mettre le modèle en mode entraînement
    optimizer.zero_grad()  # Remise à zéro des gradients
    
    # Passage avant
    output = model(x_train)
    
    # Calcul de la perte
    loss = criterion(output, y_train)
    
    # Rétropropagation et optimisation
    loss.backward()  # Calcul des gradients
    optimizer.step()  # Mise à jour des poids
    
    # Afficher la perte toutes les époques
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Tester le modèle sur un autre batch d'exemple
model.eval()  # Passer le modèle en mode évaluation
x_test = torch.randn(5, 1, 28, 28)  # Test avec 5 images aléatoires
output_test = model(x_test)
print("Sortie du modèle (test) :")
print(output_test)
