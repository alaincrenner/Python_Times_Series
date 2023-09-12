import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assurez-vous que le fichier "DataSet.xlsx" est dans le même répertoire que votre script Python
df = pd.read_excel("DataSet.xlsx")

# Tri de la colonne "date" du plus ancien au plus récent
df.sort_values(by="Date", inplace=True)

summary = df.describe()
print(summary)

date = df["Date"]
dernier = df["Dernier"]
ouvert = df["Ouv."]
#max = df["Plus Haut"]
min = df["Plus Bas"]
variation = df["Variation %"]

plt.figure(figsize=(12, 6))  # Définir la taille du graphique
plt.plot(date, dernier, marker='o', linestyle='-', color='b')
plt.xlabel("Date")
plt.ylabel("Dernier")
plt.title("Variation de Dernier au fil du temps")
plt.xticks(rotation=45)  # Rotation des étiquettes de l'axe des x pour une meilleure lisibilité
plt.grid(True)  # Ajouter une grille

plt.tight_layout()  # Pour s'assurer que les étiquettes ne se chevauchent pas
plt.show()
