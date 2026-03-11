print("\n===== Exercice 1 =====")
# exo 1 :

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

########################################################################################################

print("\n===== TP 1 =====")
# TP Descente de Gradient :

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

####################################################################################################

print("\n===== TP 2 =====")
# TP Perceptron :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

def step(s):
    return 1 if s >= 0 else 0

def perceptron_train(X, y, eta=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0.0       
    for _ in range(epochs):          
        for xi, yi in zip(X, y):  
            s = np.dot(w, xi) + b 
            y_pred = step(s)     
            error = yi - y_pred    
            if error != 0:         
                w += eta * error * xi
                b += eta * error
    return w, b

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

w_and, b_and = perceptron_train(X, y_and)
w_or,  b_or  = perceptron_train(X, y_or)
w_xor, b_xor = perceptron_train(X, y_xor)

print("AND : poids =", w_and, "biais =", b_and)
print("OR  : poids =", w_or,  "biais =", b_or)
print("XOR : poids =", w_xor, "biais =", b_xor)

def plot_decision_boundary(X, y, w, b, title):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                         np.linspace(-0.5, 1.5, 100))
    Z = np.array([step(w[0]*x_ + w[1]*y_ + b)
                  for x_, y_ in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:,0], X[:,1], c=y, s=100,
                edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.show()

plot_decision_boundary(X, y_and, w_and, b_and, "Perceptron - AND")
plot_decision_boundary(X, y_or,  w_or,  b_or,  "Perceptron - OR")
plot_decision_boundary(X, y_xor, w_xor, b_xor, "Perceptron - XOR (échoue !)")

for nom, y in [("AND", y_and), ("OR", y_or), ("XOR", y_xor)]:
    clf = Perceptron(max_iter=10, eta0=0.1)
    clf.fit(X, y)
    print(f"\nScikit-learn {nom}")
    print("Poids :", clf.coef_)
    print("Biais :", clf.intercept_)
    plot_decision_boundary(X, y, clf.coef_[0], clf.intercept_[0],
                           f"Sklearn - {nom}")

print("\nConclusion :")
print("AND et OR  → perceptron converge")
print("XOR        → perceptron échoue (pas linéairement séparable)")
print("Solution   → il faut un réseau multicouche (MLP)")


from matplotlib.animation import FuncAnimation

def animate_perceptron(X, y, title):
    w = np.zeros(X.shape[1])
    b = 0.0
    eta = 0.1
    history = []

    for _ in range(10): 
        for xi, yi in zip(X, y):
            s = np.dot(w, xi) + b
            y_pred = step(s)
            error = yi - y_pred
            if error != 0:
                w += eta * error * xi
                b += eta * error
            history.append((w.copy(), b)) 

    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                         np.linspace(-0.5, 1.5, 100))

    def update(frame):
        ax.clear()
        w_f, b_f = history[frame]
        Z = np.array([step(w_f[0]*x_ + w_f[1]*y_ + b_f)
                      for x_, y_ in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        ax.scatter(X[:,0], X[:,1], c=y, s=100,
                   edgecolors='k', cmap=plt.cm.Paired)
        ax.set_title(f"{title} — itération {frame+1}")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    ani = FuncAnimation(fig, update, frames=len(history), interval=200)
    plt.show()

animate_perceptron(X, y_and, "AND")
animate_perceptron(X, y_or,  "OR")