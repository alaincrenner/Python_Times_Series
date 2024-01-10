import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données depuis le fichier Excel
df = pd.read_excel("DataSet2.xlsx")

# Tri de la colonne "date" du plus ancien au plus récent
df.sort_values(by="Date", inplace=True)

# Extraire les colonnes d'intérêt pour l'apprentissage
date_df = df["Date"]
dernier_df = df["Dernier"]

# Convertissez les dates en valeurs numériques (nombre de jours depuis la première date)
date_df = (date_df - date_df.min()) / np.timedelta64(1, 'D')

# Créez un modèle de régression linéaire
model = LinearRegression()

# Ajustez le modèle aux données
model.fit(date_df.values.reshape(-1, 1), dernier_df)

# Réalisez des prédictions sur l'ensemble des données
predictions = model.predict(date_df.values.reshape(-1, 1))

# Visualisez les prédictions
plt.figure(figsize=(12, 6))
plt.plot(date_df, dernier_df, label="Données", color='b')
plt.plot(date_df, predictions, label="Prédictions", color='r')
plt.xlabel("Date")
plt.ylabel("Dernier")
plt.title("Prédictions de régression linéaire pour la série temporelle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Évaluez les performances du modèle (par exemple, RMSE)
rmse = np.sqrt(mean_squared_error(dernier_df, predictions))
print(f"RMSE: {rmse}")
