import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Génération de données fictives
np.random.seed(0)
revenu = np.random.uniform(20000, 100000, 50)  # Revenu en dollars
age = np.random.uniform(20, 60, 50)  # Âge en années

# Génération de la prime d'assurance avec bruit
prime = 500 + 0.05 * revenu + 20 * age + np.random.normal(0, 500, 50)

# Préparer X (les variables explicatives) et y (la variable cible)
X = np.column_stack((revenu, age))
y = prime

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédictions sur les données existantes
y_pred = model.predict(X)

# Affichage des coefficients du modèle
print(f"Intercept : {model.intercept_:.2f}")
print(f"Coefficient pour le revenu : {model.coef_[0]:.4f}")
print(f"Coefficient pour l'âge : {model.coef_[1]:.4f}")

# Graphique en 3D pour visualiser la régression
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Affichage des points réels
ax.scatter(revenu, age, prime, color='blue', label="Données réelles")

# Création d'une grille pour afficher le plan de régression
revenu_range = np.linspace(min(revenu), max(revenu), 10)
age_range = np.linspace(min(age), max(age), 10)
revenu_grid, age_grid = np.meshgrid(revenu_range, age_range)
prime_grid = (model.intercept_ + model.coef_[0] * revenu_grid + model.coef_[1] * age_grid)

# Affichage du plan de régression
ax.plot_surface(revenu_grid, age_grid, prime_grid, color='red', alpha=0.5)

# Étiquettes et titre
ax.set_xlabel("Revenu ($)")
ax.set_ylabel("Âge (années)")
ax.set_zlabel("Prime d'assurance ($)")
ax.set_title("Régression linéaire en assurance")

plt.legend()
plt.show()
