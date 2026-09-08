import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, max_it=200, lr=0.1):
        self.max_it = max_it
        self.lr = lr
        self.W = None
        self.b = 0

    def activation(self, x):
        return 1 / (1 + math.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        np.random.seed(42)
        self.W = np.random.uniform(
            -6 / math.sqrt(n_samples),
            6 / math.sqrt(n_samples),
            n_features
        )

        for epoch in range(1, self.max_it + 1):
            total_loss = 0

            for x, target in zip(X, y):
                z = np.dot(self.W, x) + self.b
                y_hat = self.activation(z)

                error = target - y_hat
                total_loss += 0.5 * error ** 2

                # Update des poids et du biais
                gradient = error * y_hat * (1 - y_hat)
                self.W += self.lr * gradient * x
                self.b += self.lr * gradient

            if epoch % 20 == 0:
                print(
                    f"Epoch {epoch:3d} -> "
                    f"Loss: {total_loss / n_samples:.6f}"
                )

    def predict_proba(self, X):
        probabilities = []

        for x in X:
            z = np.dot(self.W, x) + self.b
            probabilities.append(self.activation(z))

        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def accuracy(self, y_test, y_predicted):
        return np.mean(y_test == y_predicted)


if __name__ == "__main__":

    # Création du dataset
    X, y = datasets.make_blobs(
        n_samples=200,
        n_features=2,
        centers=2,
        cluster_std=1.5,
        random_state=2
    )

    # Séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123
    )

    # Visualisation des données
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        marker="o"
    )
    plt.title("Données d'entraînement")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Création et entraînement du modèle
    model = Perceptron(max_it=200, lr=0.1)
    model.fit(X_train, y_train)

    # Prédictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = model.accuracy(y_test, predictions)
    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    # Frontière de décision
    x_min = np.min(X_train[:, 0])
    x_max = np.max(X_train[:, 0])

    y_min = (-model.W[0] * x_min - model.b) / model.W[1]
    y_max = (-model.W[0] * x_max - model.b) / model.W[1]

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        marker="o"
    )

    plt.plot(
        [x_min, x_max],
        [y_min, y_max],
        "k-",
        label="Decision Boundary"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron - Decision Boundary")
    plt.legend()
    plt.show()
