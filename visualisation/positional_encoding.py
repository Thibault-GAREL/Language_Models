import numpy as np
import matplotlib.pyplot as plt


def generer_positional_encoding(max_seq_len, d_model):
    """
    Génère la matrice de Positional Encoding.

    Arguments:
    - max_seq_len : Le nombre maximum de tokens (positions) dans une phrase.
    - d_model : La taille du vecteur d'embedding.
    """
    # 1. Initialiser une matrice vide (zéros) de taille (max_seq_len, d_model)
    pe = np.zeros((max_seq_len, d_model))

    # 2. Créer un vecteur colonne pour les positions (de 0 à max_seq_len - 1)
    # Forme : (max_seq_len, 1)
    position = np.arange(0, max_seq_len)[:, np.newaxis]

    # 3. Créer le terme de division (le dénominateur dans la formule)
    # On utilise np.exp et le log pour une meilleure stabilité numérique,
    # c'est l'équivalent de 1 / (10000 ** (2i / d_model))
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # 4. Appliquer le Sinus sur les indices pairs (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)

    # 5. Appliquer le Cosinus sur les indices impairs (1, 3, 5...)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


# --- ZONE D'EXPÉRIMENTATION ---
# Vous pouvez modifier ces variables pour voir comment cela affecte le résultat !
D_MODEL = 128  # Taille de l'embedding (doit être un nombre pair)
MAX_SEQ_LEN = 50  # Nombre de mots/tokens dans notre phrase imaginaire

# Génération de la matrice
pe_matrix = generer_positional_encoding(MAX_SEQ_LEN, D_MODEL)

# --- VISUALISATIONS ---
plt.figure(figsize=(15, 8))

# 1. Visualisation sous forme de Heatmap (Carte de chaleur)
# Cela montre la matrice entière : les lignes sont les mots, les colonnes sont les dimensions
plt.subplot(1, 2, 1)
plt.pcolormesh(pe_matrix, cmap="RdBu", vmin=-1, vmax=1)
plt.title(f"Matrice globale (d_model={D_MODEL}, seq_len={MAX_SEQ_LEN})")
plt.xlabel("Profondeur de la dimension (0 à d_model)")
plt.ylabel("Position du token dans la phrase")
plt.colorbar(label="Valeur")
plt.gca().invert_yaxis()  # Inverser l'axe Y pour avoir la position 0 en haut

# 2. Visualisation des courbes pour quelques dimensions spécifiques
# On regarde comment la valeur évolue en fonction de la position du mot
plt.subplot(1, 2, 2)
# On trace l'évolution pour les dimensions 4, 5, 20 et 21
plt.plot(pe_matrix[:, 4], label="Dimension 4 (Sinus, fréquence haute)", color="blue")
plt.plot(pe_matrix[:, 5], label="Dimension 5 (Cosinus, fréquence haute)", color="cyan")
plt.plot(pe_matrix[:, 20], label="Dimension 20 (Sinus, fréquence basse)", color="red")
plt.plot(
    pe_matrix[:, 21], label="Dimension 21 (Cosinus, fréquence basse)", color="orange"
)

plt.title("Évolution des valeurs selon la position")
plt.xlabel("Position du token dans la phrase")
plt.ylabel("Valeur (-1 à 1)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
