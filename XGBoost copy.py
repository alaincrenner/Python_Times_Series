import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')

# Load the dataset
df = pd.read_excel("DataSet3.xlsx")
df = df[["Date", "Dernier"]]

df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Split the data into training and testing sets
# Randomly split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# future prediction:
# Création d'une liste de dates de prédictions futures
start_date = datetime(2023, 10, 18)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start_date, end_date)

# Création d'un DataFrame avec les informations requises
df_future_data = pd.DataFrame({'Date': date_range})

def create_time_feature(df):
    df['dayofmonth'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    # Use isocalendar() to get the week information
    week_info = df['Date'].dt.isocalendar()
    df['weekofyear'] = week_info.week
    return df

future_data = create_time_feature(df_future_data)
future_data = future_data.drop(['Date'], axis=1)

train = create_time_feature(train)
test = create_time_feature(test)

train = train.copy() 
test = test.copy() 

# Extracting target variable and features for the training set
y_train = train['Dernier']
X_train = train.drop(["Dernier", 'Date'], axis=1)

# Extracting target variable and features for the testing set
y_test = test['Dernier']
X_test = test.drop(["Dernier", 'Date'], axis=1)

# Fitting the model
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=False)

# Predicting on the test set
test['Dernier_Prediction'] = reg.predict(X_test)

future_prediction = reg.predict(future_data)
future_data['Dernier_Prediction'] = future_prediction
future_data['Date'] = date_range

mse = mean_squared_error(test['Dernier'], test['Dernier_Prediction'])
mae = mean_absolute_error(test['Dernier'], test['Dernier_Prediction'])
mape = mean_absolute_percentage_error(test['Dernier'], test['Dernier_Prediction'])

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Absolute Percentage Error: {mape}')

# Créer le DataFrame pour les prédictions
df_predictions = pd.DataFrame({'Date': date_range, 'Prediction': future_data['Dernier_Prediction']})

# Afficher les résultats
#print(df_predictions[['Date', 'Prediction']].sort_values(by='Date', ascending=False).to_string(index=False))
# Exporter les prédictions dans un fichier Excel
df_predictions[['Date', 'Prediction']].to_excel('predictions2.xlsx', index=False)

# Plotting the results
train['Dernier'].plot(style='k', figsize=(10, 5), label='Actual')
test['Dernier_Prediction'].plot(style='r', figsize=(10, 5), label='prediction')
plt.plot(future_data['Date'], future_data['Dernier_Prediction'], label='Future Prediction', color='blue')
plt.title('Evolution du taux de change de l euros par rapport aux dollars')
plt.legend()
plt.show()