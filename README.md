# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 18-10-2025

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
data = pd.read_csv("/content/blood_donor_dataset.csv")

data['created_at'] = pd.to_datetime(data['created_at'])

data.set_index('created_at', inplace=True)
data = data.sort_index()

numeric_data = data.select_dtypes(include=np.number)
data = numeric_data.resample('M').mean()


def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test_data))
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, test_data[target_variable], label='Actual Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()
    print("Root Mean Squared Error (RMSE):", rmse)
arima_model(data, 'number_of_donation', order=(5,1,0))
```

### OUTPUT:
<img width="1149" height="623" alt="Screenshot (7)" src="https://github.com/user-attachments/assets/4fb38ed9-895a-45ba-8e45-ab1fa5c6ccf1" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
