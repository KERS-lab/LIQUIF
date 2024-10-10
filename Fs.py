# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:32:23 2024

@author: iayoujil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# données initiales
h = float(input('Hauteur de la nappe en [m]: '))
gamma = float(input('Poids volumique au dessus de la nappe [kN/m3]: '))
a = float(input('Accélération de calcul ag [m/s²]: '))
Mw = float(input('Magnitude de référence: '))
Pa = float(input('Pression atmo [kPa]: '))

# Charger les données du fichier Excel
df = pd.read_excel('F1.xlsx', sheet_name='Feuil1')

# Calculer sigma totale
df['sigma_totale'] = gamma * df['z']

# Calculer sigma effective
def calcul_sigma_effective(z):
    if z < h:
        return z * gamma
    return h * gamma + (z - h) * (gamma - 10)

df['sigma_effective'] = df['z'].apply(calcul_sigma_effective)

# Fonction générale pour les calculs de n
def calcul_n(row, n_value):
    C, D, E, F, G = row['z'], row['qc'], row['fs'], row['sigma_totale'], row['sigma_effective']
    
    if C == "" or G <= 0:
        return np.nan

    N_part1 = (D * 1000) - F
    if N_part1 <= 0:
        return np.nan

    part1 = ((3.47 - np.log10((N_part1 / Pa) * (Pa / G)**n_value)) ** 2)
    N_part2 = 100 * E
    D_part2 = (D * 1000) - F

    if D_part2 <= 0 or N_part2 <= 0:
        return np.nan

    part2 = ((1.22 + np.log10(N_part2 / D_part2)) ** 2)
    return np.sqrt(part1 + part2)

# Appliquer les calculs de n
for n_val in [1, 0.5, 0.7]:
    df[f'n_{n_val}'] = df.apply(lambda row: calcul_n(row, n_val), axis=1)

# Calculer Ic
def IC(row):
    C, D, E, F, G = row['z'], row['qc'], row['fs'], row['sigma_totale'], row['sigma_effective']
    
    if C == "" or G <= 0:
        return np.nan

    N1 = (D * 1000) - F
    D1 = Pa * (Pa / G)

    if N1 <= 0 or D1 <= 0:
        return np.nan

    P1 = (3.47 - np.log10(N1 / D1)) ** 2
    P2 = (1.22 + np.log10(100 * E / (D * 1000 - F))) ** 2
    result1 = np.sqrt(P1 + P2)

    if result1 > 2.6:
        return result1

    P3 = (3.47 - np.log10(D * 1000 / (Pa * (Pa / G) ** 0.5))) ** 2
    result2 = np.sqrt(P3 + P2)

    if result2 > 2.6:
        return np.sqrt((3.47 - np.log(D * 1000 / (Pa * (Pa / G) ** 0.7))) ** 2 + P2)

    return np.sqrt(P3 + P2)

df['Ic'] = df.apply(IC, axis=1)

# Calculer qc1N
def qc1N(row):
    C, D, E, F, G = row['z'], row['qc'], row['fs'], row['sigma_totale'], row['sigma_effective']

    if C == "" or G <= 0:
        return np.nan

    numerator_part1 = (D * 1000) - F
    denominator_part1 = Pa * (Pa / G)

    if numerator_part1 <= 0:
        return np.nan

    part1 = (3.47 - np.log10(numerator_part1 / denominator_part1)) ** 2
    part2 = (1.22 + np.log10(100 * E / (D * 1000 - F))) ** 2
    result1 = np.sqrt(part1 + part2)

    if result1 > 2.6:
        return min(1.7, (Pa / G) ** 1) * (D * 1000 / Pa)

    part3 = (3.47 - np.log10(D * 1000 / Pa * (Pa / G) ** 0.5)) ** 2
    result2 = np.sqrt(part3 + part2)

    if result2 > 2.6:
        return min(1.7, (Pa / G) ** 0.7) * (D * 1000 / Pa)

    return min(1.7, (Pa / G) ** 0.5) * (D * 1000 / Pa)

df['qc1N'] = df.apply(qc1N, axis=1)

# Calculer Kc
def calcul_Kc(row):
    K = row['Ic']
    if K > 1.64:
        return -0.403 * K ** 4 + 5.581 * K ** 3 - 21.63 * K ** 2 + 33.75 * K - 17.88
    return 1

df['Kc'] = df.apply(calcul_Kc, axis=1)

# Calculer Cs
df['Cs'] = df['qc1N'] * df['Kc']

# Calculer CRR
def CRR(row):
    O = row['Cs']
    if O < 50:
        return 0.833 * O / 1000 + 0.05
    return 93 * (O / 1000) ** 3 + 0.08

df['CRR'] = df.apply(CRR, axis=1)

# Calculer CSR
def CSR(row):
    F, G, C = row['sigma_totale'], row['sigma_effective'], row['z']
    if G == 0:
        return np.nan
    return 0.65 * a / 9.81 * F / G * (1 - 0.015 * C)

df['CSR'] = df.apply(CSR, axis=1)

# Calculer Fs
def Fs(row):
    K, P, Q = row['Ic'], row['CRR'], row['CSR']
    MFS = 10**2.24 / Mw**2.56
    if Q == 0 or K >= 2.6:
        return np.nan
    return MFS * P / Q

df['Fs'] = df.apply(Fs, axis=1)

# Afficher toutes les lignes du DataFrame
pd.set_option('display.max_rows', None)
print(df)

# Supprimer les lignes contenant des NaN ou des Inf
df_clean = df[np.isfinite(df['Fs']) & np.isfinite(df['z'])]

# Tracer Fs en fonction de z
plt.figure(figsize=(10, 6))
plt.plot(df_clean['Fs'], df_clean['z'], linestyle='-', color='blue', markersize=8)

# Personnaliser les axes
plt.title('Coefficient de sécurité Fs')
plt.xlabel('Fs')
plt.ylabel('Profondeur m')

# Afficher l'axe des abscisses en haut
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

# Limiter l'axe des abscisses entre 0 et 5
plt.xlim(0, 5)

# Ajuster les limites de l'axe des ordonnées
plt.ylim(min(df_clean['z']), max(df_clean['z']))  # Limites de l'axe des ordonnées

# Inverser l'axe des ordonnées
plt.gca().invert_yaxis()

# Ajouter les lignes verticales pour Fs = 1 et Fs = 1.25
plt.axvline(x=1, color='red', linestyle='-', label='Fs = 1')
plt.axvline(x=1.25, color='green', linestyle='-', label='Fs = 1.25')

# Personnaliser les graduations sur l'axe des abscisses
ticks = np.arange(0, 5.5, 0.5)  # Définir les positions des ticks
plt.xticks(ticks)  # Appliquer les ticks définis

# Modifier la couleur des ticks spécifiques (Fs=1 et Fs=1.25)
ax = plt.gca()
for tick in ax.get_xticklabels():
    if tick.get_text() == '1.0':
        tick.set_color('red')
    elif tick.get_text() == '1.25':
        tick.set_color('green')

# Afficher la légende
plt.legend()

# Afficher la grille
plt.grid()

# Sauvegarder le graphique en tant qu'image
graph_image_path = 'Fs_graphique.png'
plt.savefig(graph_image_path)

# Afficher le graphique
plt.show()

# Exporter les résultats dans une nouvelle feuille Excel
output_path = 'C:/Users/IAYOUJIL/OneDrive - FAYAT/Bureau/PYTHON/Liquifaction/F1.xlsx'
with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
    df_clean.to_excel(writer, sheet_name='resultat', index=False)

    # Insérer le graphique dans la feuille "Fs"
    workbook = writer.book
    worksheet = workbook.create_sheet(title='Fs')
    
    # Insérer l'image
    from openpyxl.drawing.image import Image
    img = Image(graph_image_path)
    worksheet.add_image(img, 'A3')  # Ajouter l'image à la cellule A1

# Supprimer l'image temporaire si nécessaire
import os
os.remove(graph_image_path)

# Créer un fichier PDF pour stocker les résultats
with PdfPages('resultats.pdf') as pdf:
    # 3. Ajouter le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(df_clean['Fs'], df_clean['z'], linestyle='-', color='blue', markersize=8)
    
    # Personnaliser les axes
    plt.title('Coefficient de sécurité Fs')
    plt.xlabel('Fs')
    plt.ylabel('Profondeur m')

    # Afficher l'axe des abscisses en haut
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')

    # Limiter l'axe des abscisses entre 0 et 5
    plt.xlim(0, 5)

    # Ajuster les limites de l'axe des ordonnées
    plt.ylim(min(df_clean['z']), max(df_clean['z']))  # Limites de l'axe des ordonnées

    # Inverser l'axe des ordonnées
    plt.gca().invert_yaxis()

    # Ajouter les lignes verticales pour Fs = 1 et Fs = 1.25
    plt.axvline(x=1, color='red', linestyle='-', label='Fs = 1')
    plt.axvline(x=1.25, color='green', linestyle='-', label='Fs = 1.25')

    # Personnaliser les graduations sur l'axe des abscisses
    ticks = np.arange(0, 5.5, 0.5)  # Définir les positions des ticks
    plt.xticks(ticks)  # Appliquer les ticks définis

    # Modifier la couleur des ticks spécifiques (Fs=1 et Fs=1.25)
    ax = plt.gca()
    for tick in ax.get_xticklabels():
        if tick.get_text() == '1.0':
            tick.set_color('red')
        elif tick.get_text() == '1.25':
            tick.set_color('green')

    # Afficher la légende
    plt.legend()

    # Afficher la grille
    plt.grid()

    # Afficher le graphique
    pdf.savefig()  # Sauvegarder le graphique
    plt.close()
   
print("Le fichier PDF a été créé avec succès.")