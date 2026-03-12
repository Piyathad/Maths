print("\n===== TP 1 =====")
# tp 1 :

import numpy as np
import matplotlib.pyplot as plt

# ── TABLEAU XOR ──
# x1  x2  y
#  0   0   0
#  0   1   1
#  1   0   1
#  1   1   0
# XOR = 1 seulement quand x1 et x2 sont différents

# ── DONNÉES XOR ──
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])

def step(s):
    return 1 if s >= 0 else 0

# ── ÉTAPE 1 : Initialiser le perceptron ──
w = np.zeros(2) 
b = 0.0  
eta = 0.1  
epochs = 10  

# ── ÉTAPE 2 : Entraîner le perceptron ──
for epoch in range(epochs):
    nb_erreurs = 0
    for xi, yi in zip(X, y):
        s = np.dot(w, xi) + b
        y_pred = step(s)
        error = yi - y_pred
        if error != 0:
            w += eta * error * xi
            b += eta * error
            nb_erreurs += 1
    print(f"Epoch {epoch+1} | erreurs: {nb_erreurs} | w={w} | b={b}")

print("\nPoids finaux:", w)
print("Biais final:", b)

# ── ÉTAPE 3 : Analyse graphique ──
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
Z = np.array([step(w[0]*x_ + w[1]*y_ + b)
              for x_, y_ in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=y, s=100, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Perceptron XOR — aucune droite ne sépare les points !")
plt.show()

# ── QUESTIONS ──

# Q1 — Pourquoi le perceptron échoue sur XOR ?
# Les erreurs ne descendent jamais à 0 (4 erreurs en boucle). XOR n'est pas linéairement séparable → une seule droite ne peut pas séparer les points bleus et oranges qui sont en diagonale croisée.

# Q2 — Que se passe-t-il si on augmente les itérations ou le taux d'apprentissage ?
# Plus d'itérations → ça ne change rien, les erreurs restent à 4, ça tourne en boucle infinie
# Changer le taux d'apprentissage → ça ne résout pas le problème, juste des poids différents mais toujours des erreurs

# Q3 — Comment une couche cachée permet de résoudre XOR ?
# Une couche cachée transforme les données dans un nouvel espace où elles deviennent séparables. Par exemple :
# Couche 1 apprend AND et OR séparément
# Couche 2 combine les deux pour faire XOR
