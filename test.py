import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', ConvergenceWarning)

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 9)
mpl.rcParams['axes.grid'] = True

from datetime import datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.optimize import curve_fit
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


if __name__ == '__main__':
	dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
	data = pd.read_csv('NFLX.csv', parse_dates=['Date'], date_parser=dateparse)
	
	data = data.drop(columns=["Open", "Low", "High", "Adj Close", "Volume"])

	x, y = data["Date"], data["Close"]
	print("x, y")

	train_data, test_data = np.split(data, [int(.94 * len(data))])
	X_train, y_train = train_data["Date"], train_data["Close"]
	X_test, y_test = test_data["Date"], test_data["Close"]

	print("train, test")

	errors_aic_ts = None
	errors_bic_ts = None
	errors_mse_ts = None

	if os.path.isfile("best_params_aic_ts.csv") and os.path.isfile("best_params_bic_ts.csv") and os.path.isfile("best_params_mse_ts.csv"):
		errors_aic_ts = pd.read_csv("best_params_aic_ts.csv")
		errors_bic_ts = pd.read_csv("best_params_bic_ts.csv")
		errors_mse_ts = pd.read_csv("best_params_mse_ts.csv")
	else:
		errors_aic_ts = pd.DataFrame(columns=["p", "d", "q", "P", "D", "Q", "season", "value"])
		errors_bic_ts = pd.DataFrame(columns=["p", "d", "q", "P", "D", "Q", "season", "value"])
		errors_mse_ts = pd.DataFrame(columns=["p", "d", "q", "P", "D", "Q", "season", "value"])
		for p in range(10):
			for q in range(10):
				for P in range(10):
					for Q in range(10):
						model = ARIMA(y_train, order=(p, 1, q), seasonal_order=(P, 1, Q, 5))
						fitted = model.fit()

						errors_aic_ts = errors_aic_ts.append({"p": p, "d": 1, "q": q, "P": P, "D": 1, "Q": Q, "season": 5, "value": fitted.aic}, ignore_index=True)
						errors_bic_ts = errors_bic_ts.append({"p": p, "d": 1, "q": q, "P": P, "D": 1, "Q": Q, "season": 5, "value": fitted.bic}, ignore_index=True)
						errors_mse_ts = errors_mse_ts.append({"p": p, "d": 1, "q": q, "P": P, "D": 1, "Q": Q, "season": 5, "value": fitted.mse}, ignore_index=True)

						print(f'p={p}, d=1, q={q} P={P} D=1 Q={Q}/ AIC: {fitted.aic}')
						print(f'p={p}, d=1, q={q} P={P} D=1 Q={Q}/ BIC: {fitted.bic}')
						print(f'p={p}, d=1, q={q} P={P} D=1 Q={Q}/ MSE: {fitted.mse}')
						print("--------------------------------------------")
		errors_aic_ts.to_csv("best_params_aic_ts.csv", index=False)
		errors_bic_ts.to_csv("best_params_bic_ts.csv", index=False)
		errors_mse_ts.to_csv("best_params_mse_ts.csv", index=False)