print("\n===== TP 1 =====")
# tp 2 :

import numpy as np

X = np.array([50, 60, 70, 80, 90])
Y = np.array([100, 120, 140, 160, 180])
X_matrix = np.column_stack((np.ones(len(X)), X))
print("Matrice X :")
print(X_matrix, "\n")

theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y

b = abs(theta[0])
a = theta[1]

print("pente a =", round(a, 2))
print("biais b =", round(b, 2))
print(f"Equation finale : y = {round(a, 2)}x + {round(b, 2)}")


print("\nInterprétation :")
print("surface 50m² → prix", round(a*50 + b, 2), "k€")
print("surface 60m² → prix", round(a*60 + b, 2), "k€")
print("surface 70m² → prix", round(a*70 + b, 2), "k€")
print("surface 80m² → prix", round(a*80 + b, 2), "k€")
print("surface 90m² → prix", round(a*90 + b, 2), "k€")
print("\nConclusion : chaque +1m² = +2 000€ de prix.")
print("La droite y = 2x correspond aux données")

###############################################

print("\n===== TP 3 =====")
# tp 3 :

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
Y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
n = len(X)

w = 0
b = 0
lr = 0.001
iterations = 1000
loss_history = []

for i in range(iterations):
    y_pred = w * X + b
    loss = np.mean((Y - y_pred)**2)
    loss_history.append(loss)
    dw = (-2/n) * np.sum(X * (Y - y_pred))
    db = (-2/n) * np.sum(Y - y_pred)
    w = w - lr * dw
    b = b - lr * db

print("Gradient Descent")
print("w =", round(w, 2))
print("b =", round(b, 2))
print("Equation : y =", round(w, 2), "x +", round(b, 2))

model = LinearRegression()
model.fit(X.reshape(-1, 1), Y)
print("\nScikit-learn")
print("coef =", round(model.coef_[0], 2))
print("intercept =", round(model.intercept_, 2))

print("\nAnalyse de divergence :")
print("1 - Learning rate trop élevé (ex: lr = 0.5)")
print("   → oscillation et explosion de la loss")
print("2 - Données non normalisées (valeurs trop grandes)")
print("   → pas de stabilité numérique")

print("\nConclusion :")
print("La descente de gradient dépend de :")
print("   - learning rate")
print("   - normalisation")
print("   - initialisation des paramètres")

plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Courbe de convergence - Dataset 1")
plt.show()