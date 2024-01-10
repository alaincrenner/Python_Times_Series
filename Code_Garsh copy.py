import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Charger les données depuis le fichier Excel
df = pd.read_excel("DataSet2.xlsx")

# Tri de la colonne "date" du plus ancien au plus récent
df.sort_values(by="Date", inplace=True)

# Extraire les colonnes d'intérêt pour l'apprentissage
train = df[df["Date"] < pd.to_datetime("2023-01-01", format='%Y-%m-%d')]
test = df[df["Date"] >= pd.to_datetime("2023-01-01", format='%Y-%m-%d')]

# Modélisation GARCH(1, 1)
model = arch_model(train["Dernier"], vol='Garch', p=1, q=1)

# Ajustez le modèle aux données d'apprentissage
results = model.fit()

# Réalisez des prédictions sur les données d'apprentissage
forecast = results.conditional_volatility

# Visualisez les prédictions de volatilité
plt.figure(figsize=(12, 6))
plt.plot(train["Date"], train["Dernier"], label="Valeurs réelles", color='b')  # Ajoutez les valeurs réelles
plt.plot(train["Date"], forecast, label="Prédictions de volatilité GARCH", color='r')
plt.xlabel("Date")
plt.ylabel("Dernier")
plt.title("Prédictions de la variable 'Dernier' avec modèle GARCH(1,1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Affichez les paramètres estimés du modèle GARCH
print(results.summary())

from sklearn.metrics import mean_squared_error

# Réalisez des prédictions sur les données d'apprentissage
forecast = results.conditional_volatility

# Calculez le MSE
mse = mean_squared_error(train["Dernier"], forecast)
print(f"MSE du modèle GARCH(1,1) pour la variable 'Dernier': {mse}")
