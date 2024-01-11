# Import des packages
import pandas_datareader.data as web
import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error

# Récupération des données
df = pd.read_excel("DataSet3.xlsx")
df = df[["Date", "Dernier"]]
df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
del df['Date']
sns.set()

# Séparation des données en entrainement et test
train = df[df.index < pd.to_datetime("2023-07-01", format='%Y-%m-%d')]
test = df[df.index > pd.to_datetime("2023-07-01", format='%Y-%m-%d')]

y = train['Dernier']

#Implémentation de ARMA
ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit()
y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out_ARMAmodel = y_pred_df["Predictions"] 

#Implémentation de ARIMA
ARIMAmodel = ARIMA(y, order = (4, 2, 3))
ARIMAmodel = ARIMAmodel.fit()
y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out_ARIMAmodel = y_pred_df["Predictions"] 

#Implémentation de SARIMA
SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()
y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out_SARIMAXmodel = y_pred_df["Predictions"] 

# Configuration du plot
plt.ylabel('Dernier')
plt.xlabel('Date')
plt.xticks(rotation=45)

# Plot prédiction
plt.plot(y_pred_out_ARMAmodel, color='orange', label='ARMA predictions')
plt.plot(y_pred_out_ARIMAmodel, color='yellow', label='ARIMA predictions')
plt.plot(y_pred_out_SARIMAXmodel, color='blue', label='SARIMA predictions')

# Plot data
plt.plot(train, color='black', label='Train data')
plt.plot(test, color='red', label='Test data')

# Ajour de la légende
plt.legend()
plt.title("Train/Test split for Dernier data")

# Affichage du graphique
plt.show()