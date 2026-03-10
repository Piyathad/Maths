print("\n===== EXERCICE 1 =====")
# exo 1 :

v1 = [1,3,5]
v2 = [-2, -8, 13]
v_add = [v1[i] + v2[i] for i in range(len(v1))]
print(v_add)

#################################################

print("\n===== EXERCICE 2 =====")
# exo 2 : 

print('-- VECTEUR --')
v1 = [1, 2, 3]
v2 = [2, 3, 4]
v3 = [[1, 2], [3, 4]]
v4 = [3, 4]
print("v1:", v1)
print("v2:", v2)
print("v3:", v3)
print("v4:", v4, "\n")

print('-- ADDITION VECTEUR --')
somme = [v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]]
print(somme, "\n")

print('-- MULTIPLICATION PAR UN SCALAIRE --')
scalaire = [v1[0]*2, v1[1]*2, v1[2]*2]
print(scalaire, "\n")

print('-- PRODUIT SCALAIRE (DOT)--')
dot = (v1[0]*v2[0]) + (v1[1]*v2[1]) + (v1[2]*v2[2])
print(dot, "\n")

print('-- PRODUIT MATRICE VECTEUR --')
a = (v3[0][0]*v4[0]) + (v3[0][1]*v4[1])
x = (v3[1][0]*v4[0]) + (v3[1][1]*v4[1])
print("Résultat :", [a, x], "\n")

##################################################

print("\n===== EXERCICE 3 =====")
# exo 3 : 

x = [50, 70, 90]
y = [100, 140, 180]

x_moy = (50 + 70 + 90) / 3
y_moy = (100 + 140 + 180) / 3
print("Moyenne x:", x_moy)
print("Moyenne y:", y_moy)

num = (50-x_moy)*(100-y_moy) + (70-x_moy)*(140-y_moy) + (90-x_moy)*(180-y_moy)
print("Numérateur:", num)

den = (50-x_moy)**2 + (70-x_moy)**2 + (90-x_moy)**2
print("Dénominateur:", den)

a = num / den
b = y_moy - a * x_moy
print("a:", a)
print("b:", b)

print("Equation finale : y =", a, "x +", b)

###################################################

print("\n===== EXERCICE 4 =====")
# exo 4 :

y_reel = [100, 140, 180]
y_pred = [2*50, 2*70, 2*90]
n = 3

mse = ((y_reel[0]-y_pred[0])**2 + (y_reel[1]-y_pred[1])**2 + (y_reel[2]-y_pred[2])**2) /n

print("y réel:" , y_reel)
print("y predit:", y_pred)
print("MSE:", mse)

#################################################

print("\n===== TP 1 =====")
# exo 5 : 

import matplotlib.pyplot as plt

noms = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]

liste_x = [
    [10,8,13,9,11,14,6,4,12,7,5],
    [10,8,13,9,11,14,6,4,12,7,5],
    [10,8,13,9,11,14,6,4,12,7,5],
    [8,8,8,8,8,8,8,19,8,8,8]
]

liste_y = [
    [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68],
    [9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,7.26,7.26,4.74],
    [7.46,6.77,12.74,7.11,8.81,8.84,6.08,5.39,8.15,6.40,5.73],
    [6.58,5.76,5.76,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89]
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for k in range(4):
    x, y, n = liste_x[k], liste_y[k], 11

    x_moy = sum(x) / n
    y_moy = sum(y) / n
    a = sum((x[i]-x_moy)*(y[i]-y_moy) for i in range(n)) / sum((x[i]-x_moy)**2 for i in range(n))
    b = y_moy - a * x_moy
    mse = sum((y[i]-(a*x[i]+b))**2 for i in range(n)) / n

    print(f"── {noms[k]} ──")
    print("Moyenne x:", round(x_moy,2), "| Moyenne y:", round(y_moy,2))
    print("a =", round(a,2), "| b =", round(b,2))
    print("MSE :", round(mse,2), "\n")

    axes[k].scatter(x, y, color="blue")
    droite = [a*xi+b for xi in x]
    axes[k].plot(sorted(x), sorted(droite), color="red")
    axes[k].set_title(noms[k])
    axes[k].set_xlabel("X")
    axes[k].set_ylabel("Y")

plt.tight_layout()
plt.show()