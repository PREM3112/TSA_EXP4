# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 13/05/2026
## System Required: Climate.csv colab



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
Name: PREM R
Reg No: 212223240124
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('DailyDelhiClimateTrain.csv')

# Convert date column
data['date'] = pd.to_datetime(data['date'])

# Select column (you can change this if needed)
X = data['meantemp'].dropna()

# Set figure size
plt.rcParams['figure.figsize'] = [12, 6]

# Plot original data
plt.plot(X)
plt.title('Original Data (Mean Temperature)')
plt.show()

# ACF and PACF of original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# -------------------------------
# ARMA(1,1)
# -------------------------------

N = 1000

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title("ACF - ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1)
plt.title("PACF - ARMA(1,1)")
plt.show()

# -------------------------------
# ARMA(2,2)
# -------------------------------

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title("ACF - ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2)
plt.title("PACF - ARMA(2,2)")
plt.show()

```

# OUTPUT:
<img width="1126" height="602" alt="image" src="https://github.com/user-attachments/assets/7f66202d-5ee9-432f-bc38-1f3b71ce0356" />
<img width="1117" height="598" alt="image" src="https://github.com/user-attachments/assets/db8419f2-c404-4501-9652-3ffc55f0a51d" />
<img width="1125" height="606" alt="image" src="https://github.com/user-attachments/assets/cd9d1075-51d1-423a-836d-de454ceb1090" />
<img width="1115" height="595" alt="image" src="https://github.com/user-attachments/assets/d6466162-b051-45fe-bb04-9965ec55bae5" />
<img width="1133" height="600" alt="image" src="https://github.com/user-attachments/assets/b34c0701-5917-4e16-ba6b-2c5170fe70c5" />
<img width="1126" height="592" alt="image" src="https://github.com/user-attachments/assets/f331bceb-513e-42b5-93a4-26b4f4c5278f" />
<img width="1338" height="670" alt="image" src="https://github.com/user-attachments/assets/6d60f5e5-f370-431e-95af-312bcfe5fa61" />
<img width="1093" height="590" alt="image" src="https://github.com/user-attachments/assets/eb9107ad-1846-42bb-9d74-5cff432dab2f" />



# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
