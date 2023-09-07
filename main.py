import pandas as pd
import matplotlib.pyplot as plt

#Trier la colonne date

# Récupération des données
data = pd.read_csv("EUR_USD.csv", sep=",", decimal=".")

# Suppression de la colonne vide
data.drop("Vol.", axis=1, inplace=True)

# Vérifier s'il y a des données manquantes dans chaque colonne
missing_data = data.isnull().sum()
print(missing_data)

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

# Extraire les années et les mois
data['Année'] = data['Date'].dt.year
data['Mois'] = data['Date'].dt.month

# Créer un graphique par année
years = data['Année'].unique()
for year in years:
    year_data = data[data['Année'] == year]
    plt.figure(figsize=(12, 6))
    plt.plot(year_data['Date'], year_data['Dernier'], label=f'Taux de change USD/EUR en {year}')
    plt.xlabel('Date')
    plt.ylabel('Taux de change')
    plt.title(f'Évolution du taux de change USD/EUR en {year}')
    plt.legend()
    #plt.show()

# Créer un graphique par mois pour une année donnée (par exemple, 2022)
selected_year = 2022
year_data = data[data['Année'] == selected_year]
months = year_data['Mois'].unique()
for month in months:
    month_data = year_data[year_data['Mois'] == month]
    plt.figure(figsize=(12, 6))
    plt.plot(month_data['Date'], month_data['Dernier'], label=f'Taux de change USD/EUR en {selected_year}-{month:02d}')
    plt.xlabel('Date')
    plt.ylabel('Taux de change')
    plt.title(f'Évolution du taux de change USD/EUR en {selected_year}-{month:02d}')
    plt.legend()
    #plt.show()

# Définir la proportion de données à utiliser pour l'ensemble de test
test_size = 0.2

# Calculer l'indice de séparation entre les ensembles d'entraînement et de test
split_index = int(len(data) * (1 - test_size))

# Diviser les données en ensembles d'entraînement et de test
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

print(train_data)
print(test_data)