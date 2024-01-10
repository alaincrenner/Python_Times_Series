import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

# Assurez-vous que le fichier "DataSet.xlsx" est dans le même répertoire que votre script Python
df = pd.read_excel("DataSet2.xlsx")

# Tri de la colonne "date" du plus ancien au plus récent
df.sort_values(by="Date", inplace=True)

summary = df.describe()
print(summary)

# Calcul de la skewness (mesure de l'asymétrie)
skewness = df['Dernier'].skew()
print(f'Skewness (Asymétrie) : {skewness}')

# Calcul de la kurtosis (mesure de l'aplatissement)
kurtosis = df['Dernier'].kurtosis()
print(f'Kurtosis (Aplatissement) : {kurtosis}')

date = df["Date"]
dernier = df["Dernier"]
ouvert = df["Ouv."]
# max = df["Plus Haut"]
min = df["Plus Bas"]

# Ajoutez une courbe de tendance (ajustement polynomial)
degree = 3  # Degré du polynôme
coefficients = np.polyfit(date.index, dernier, degree)
polynomial = np.poly1d(coefficients)
trend_line = polynomial(date.index)

plt.figure(figsize=(12, 6))  # Définir la taille du graphique
plt.plot(date, dernier, marker='o', linestyle='-', color='b', label="Dernier")
plt.plot(date, trend_line, linestyle='--', color='r', label="Courbe de tendance")
plt.xlabel("Date")
plt.ylabel("Dernier")
plt.title("Variation de Dernier au fil du temps avec Courbe de Tendance")
plt.xticks(rotation=45)  # Rotation des étiquettes de l'axe des x pour une meilleure lisibilité
plt.grid(True)  # Ajouter une grille
plt.legend()  # Ajouter une légende

plt.tight_layout()  # Pour s'assurer que les étiquettes ne se chevauchent pas
plt.show()



# Tracez un histogramme de la variable "Dernier"
plt.figure(figsize=(12, 6))
plt.hist(dernier, bins=20, color='purple', edgecolor='k', density=True)  # Utilisez density=True pour obtenir une distribution de probabilité
plt.xlabel("Dernier")
plt.ylabel("Fréquence")
plt.title("Histogramme de Dernier")

# Ajustez une distribution normale aux données
mu, std = stats.norm.fit(dernier)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)

# Tracez la courbe représentant la loi normale
plt.plot(x, p, 'k', linewidth=2, label='Loi Normale')

plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Effectuer le test de Dickey-Fuller augmenté (ADF) pour la stationnarité
adf_result = sm.tsa.adfuller(dernier, autolag='AIC')

# Afficher les résultats du test ADF
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Valeurs critiques:')
for key, value in adf_result[4].items():
    print(f'{key}: {value}')

# Interprétation des résultats du test ADF
if adf_result[1] <= 0.05:
    print("Le test ADF suggère que les données sont stationnaires (p-value <= 0.05).")
else:
    print("Le test ADF suggère que les données ne sont pas stationnaires (p-value > 0.05).")





