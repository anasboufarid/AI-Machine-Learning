import numpy as np

# Fonction d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Initialisation des poids
np.random.seed(42)
w = np.random.uniform(-6 / np.sqrt(3), 6 / np.sqrt(3), (1, 2))
b = 0

# Données d'entraînement
x = np.array([
    [2, 0],
    [0, 3],
    [3, 0],
    [1, 1]
])

y = np.array([1, 0, 0, 1])

# Paramètres d'apprentissage
max_it = 10000
learning_rate = 0.01

LOSS = []
ERROR = []

# Entraînement
for epoch in range(1, max_it + 1):
    error_s = 0

    for i in range(len(x)):

        # Forward propagation
        z = np.dot(w, x[i]) + b
        y_hat = sigmoid(z)

        # Calcul de l'erreur
        error = y[i] - y_hat
        error_s += 0.5 * error ** 2

        # Mise à jour des poids et du biais
        w += learning_rate * error * (1 - y_hat) * x[i]
        b += learning_rate * error * (1 - y_hat)

    LOSS.append(error_s)
    ERROR.append(error_s * 0.25)

    # Affichage périodique
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss : {error_s:.6f}")


# Test du modèle
test = np.array([2, 0])
proba = sigmoid(np.dot(w, test) + b)

print("\n--- Résultat final ---")
print("Poids :", w)
print("Bias :", b)
print("Probabilité pour [2, 0] :", float(proba))
