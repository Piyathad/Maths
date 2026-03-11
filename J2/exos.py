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
