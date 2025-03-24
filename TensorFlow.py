import tensorflow as tf
import tensorflow_probability as t
import numpy as np

# Définir un modèle bayésien
class BayesianNN(tf.keras.Model):
    
    def __init__(self):
        super(BayesianNN, self).__init__()
        self.dense1 = t.layers.DenseFlipout(128, activation='relu')
        self.dense2 = t.layers.DenseFlipout(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Données d'exemple
x_data = np.random.randn(100, 1).astype(np.float32)
y_data = 3 * x_data + np.random.randn(100, 1).astype(np.float32) * 0.1

# Créer un modèle
model = BayesianNN()

# Compile le modèle
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
model.fit(x_data, y_data, epochs=100)

# Faire des prédictions
predictions = model(x_data)
print(predictions)