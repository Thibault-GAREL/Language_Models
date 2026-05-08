"""
Étape 2 : Visualisation de la LayerNorm en 3D.

LayerNorm normalise CHAQUE POINT individuellement à travers ses features
(contrairement à la BatchNorm qui normalise chaque feature à travers le batch).

Pour un point x = (x1, x2, x3) :
    μ_point = mean(x1, x2, x3)
    σ_point = std(x1, x2, x3)
    y_i = (x_i - μ_point) / sqrt(σ_point² + ε)

Géométriquement, pour des points en 3D, la LayerNorm :
  1. Recentre chaque point sur l'hyperplan x + y + z = 0
     (parce que la moyenne de ses coordonnées devient 0)
  2. Normalise sa distance à l'origine à √d = √3 ≈ 1.732
     (parce que la variance de ses coordonnées devient 1)

Conséquence : tous les points sortent sur un CERCLE
(intersection de la sphère de rayon √3 et du plan x+y+z=0).
"""

import numpy as np
import matplotlib.pyplot as plt


# Reproductibilité
np.random.seed(42)

# Paramètres : nuage volontairement décentré et anisotrope pour voir l'effet
N_POINTS = 500
EPS = 1e-8
D = 3  # nombre de features

# Génération du nuage X
points_X = np.random.randn(N_POINTS, D)

MEAN = np.mean(points_X, axis=0)
STD = np.std(points_X, axis=0)

# ----------------------------------------------------------------------
# LayerNorm : moyenne et écart-type calculés POUR CHAQUE POINT (axis=1)
# ----------------------------------------------------------------------
mean_per_point = points_X.mean(axis=1, keepdims=True)   # shape (N, 1)
var_per_point = points_X.var(axis=1, keepdims=True)     # shape (N, 1)

points_Y = (points_X - mean_per_point) / np.sqrt(var_per_point + EPS)


# ----------------------------------------------------------------------
# Vérifications : les propriétés attendues de la LayerNorm
# ----------------------------------------------------------------------
print("=== Avant LayerNorm (X) ===")
print(f"  Moyenne par axe : {points_X.mean(axis=0).round(3)}")
print(f"  Écart-type par axe : {points_X.std(axis=0).round(3)}")

print("\n=== Après LayerNorm (Y) ===")
print(f"  Moyenne par point (doit être ≈ 0) : "
      f"min={points_Y.mean(axis=1).min():.2e}, "
      f"max={points_Y.mean(axis=1).max():.2e}")
print(f"  Écart-type par point (doit être ≈ 1) : "
      f"min={points_Y.std(axis=1).min():.4f}, "
      f"max={points_Y.std(axis=1).max():.4f}")
print(f"  Norme par point (doit être ≈ √3 = {np.sqrt(D):.4f}) : "
      f"min={np.linalg.norm(points_Y, axis=1).min():.4f}, "
      f"max={np.linalg.norm(points_Y, axis=1).max():.4f}")
print(f"  Somme des coordonnées par point (doit être ≈ 0) : "
      f"max abs = {np.abs(points_Y.sum(axis=1)).max():.2e}")


# ----------------------------------------------------------------------
# Point mis en évidence : on suit UN point au hasard à travers la LayerNorm
# ----------------------------------------------------------------------
HIGHLIGHT_IDX = np.random.randint(0, N_POINTS)
point_X_highlight = points_X[HIGHLIGHT_IDX]
point_Y_highlight = points_Y[HIGHLIGHT_IDX]

print(f"\n=== Point suivi (index {HIGHLIGHT_IDX}) ===")
print(f"  Avant LayerNorm : {point_X_highlight.round(3)}")
print(f"  Après LayerNorm : {point_Y_highlight.round(3)}")
print(f"  Norme avant : {np.linalg.norm(point_X_highlight):.3f}")
print(f"  Norme après : {np.linalg.norm(point_Y_highlight):.3f} (≈ √3)")


# ----------------------------------------------------------------------
# Affichage côte à côte : X (avant) vs Y (après LayerNorm)
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))

# --- Sous-figure 1 : nuage X brut ---
ax1 = fig.add_subplot(121, projection="3d")
distances_X = np.linalg.norm(points_X - MEAN, axis=1)

ax1.scatter(
    points_X[:, 0], points_X[:, 1], points_X[:, 2],
    c=distances_X, cmap="viridis", s=10, alpha=0.6,
)
ax1.scatter(*MEAN, color="red", s=120, marker="X", label=f"μ = {MEAN.round(2).tolist()}")

# Point mis en évidence (vert) + flèche depuis l'origine
ax1.scatter(*point_X_highlight, color="lime", s=200, marker="o",
            edgecolors="black", linewidths=1.5, label=f"Point #{HIGHLIGHT_IDX}", zorder=5)
ax1.quiver(0, 0, 0,
           point_X_highlight[0], point_X_highlight[1], point_X_highlight[2],
           color="green", arrow_length_ratio=0.08, linewidth=2)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title(f"Avant LayerNorm\nNuage gaussien ({N_POINTS} points)")
ax1.legend()


# --- Sous-figure 2 : nuage Y après LayerNorm ---
ax2 = fig.add_subplot(122, projection="3d")
distances_Y = np.linalg.norm(points_Y, axis=1)

ax2.scatter(
    points_Y[:, 0], points_Y[:, 1], points_Y[:, 2],
    c=distances_Y, cmap="coolwarm", s=10, alpha=0.6,
)
ax2.scatter(0, 0, 0, color="red", s=120, marker="X", label="Origine")

# Même point mis en évidence (vert) + flèche depuis l'origine
ax2.scatter(*point_Y_highlight, color="lime", s=200, marker="o",
            edgecolors="black", linewidths=1.5, label=f"Point #{HIGHLIGHT_IDX}", zorder=5)
ax2.quiver(0, 0, 0,
           point_Y_highlight[0], point_Y_highlight[1], point_Y_highlight[2],
           color="green", arrow_length_ratio=0.08, linewidth=2)

# Trace l'hyperplan x+y+z=0 pour visualiser la projection
r = np.sqrt(D) * 1.2
xx, yy = np.meshgrid(np.linspace(-r, r, 10), np.linspace(-r, r, 10))
zz = -xx - yy
ax2.plot_surface(xx, yy, zz, alpha=0.1, color="gray")

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title(
    f"Après LayerNorm\nCercle dans le plan x+y+z=0, rayon √3 ≈ {np.sqrt(D):.2f}"
)
ax2.legend()

# Mêmes limites pour bien voir la forme circulaire
lim = np.sqrt(D) * 1.3
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)
ax2.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
