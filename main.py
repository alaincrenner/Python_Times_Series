import pandas as pd
import matplotlib.pyplot as plt

#Trier la colonne date

# Récupération des données
data = pd.read_excel("DataSet.xlsx")

# Vérifier s'il y a des données manquantes dans chaque colonne
missing_data = data.isnull().sum()
print(missing_data)

# Tri de la colonne "date" du plus ancien au plus récent
data.sort_values(by="Date", inplace=True)
print(data)

summary = data.describe()
print(summary)

date = data["Date"]
dernier = data["Dernier"]
ouvert = data["Ouv."]
#max = data["Plus Haut"]
min = data["Plus Bas"]
variation = data["Variation %"]

plt.figure(figsize=(12, 6))  # Définir la taille du graphique
plt.plot(date, dernier, marker='o', linestyle='-', color='b')
plt.xlabel("Date")
plt.ylabel("Dernier")
plt.title("Variation de Dernier au fil du temps")
plt.xticks(rotation=45)  # Rotation des étiquettes de l'axe des x pour une meilleure lisibilité
plt.grid(True)  # Ajouter une grille

plt.tight_layout()  # Pour s'assurer que les étiquettes ne se chevauchent pas
plt.show()

# Affichage des informations
print(data.head())

# Calculer des statistiques descriptives
descriptive_stats = data.describe()
print(descriptive_stats)

# Supprimer les espaces des milliers et remplacer les virgules par des points pour les colonnes numériques
columns_to_convert = ['Dernier', 'Ouv.', 'Plus Haut', 'Plus Bas']
for col in columns_to_convert:
    data[col] = data[col].str.replace(' ', '').str.replace(',', '.').astype(float)

print(data.head())

# Convertir la colonne "Date" en type datetime en spécifiant le format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Exemple : Tracé d'un graphique temporel
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Dernier'], label='Taux de change USD/EUR')
plt.xlabel('Date')
plt.ylabel('Taux de change')
plt.title('Évolution du taux de change USD/EUR au fil du temps')
plt.legend()
plt.show()


# Définir la proportion de données à utiliser pour l'ensemble de test
test_size = 0.2

# Calculer l'indice de séparation entre les ensembles d'entraînement et de test
split_index = int(len(data) * (1 - test_size))

# Diviser les données en ensembles d'entraînement et de test
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

print(train_data)
print(test_data)