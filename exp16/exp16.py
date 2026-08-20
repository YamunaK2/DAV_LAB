"""
Experiment 16: Time Series Analysis

AIM:
To perform Time Series Analysis on diabetes-related data, identifying trends, seasonality, smoothing, and forecasting glucose levels.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

def run_experiment_16():
    base=os.path.dirname(os.path.abspath(__file__))
    df=pd.read_csv(os.path.join(base,'diabetes9.csv'),parse_dates=['Date'])
    df=df.set_index('Date')
    print(df.head())
    plt.figure(figsize=(12,5)); plt.plot(df['Glucose'],label='Glucose Level'); plt.xlabel('Date'); plt.ylabel('Glucose Level'); plt.title('Time Series of Glucose Levels'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(base,'glucose_time_series.png'),dpi=150); plt.close()
    # Use period 7 for daily data and avoid invalid decomposition for short samples.
    decomposition=seasonal_decompose(df['Glucose'],model='additive',period=7)
    fig,axes=plt.subplots(3,1,figsize=(12,8)); decomposition.trend.plot(ax=axes[0],title='Trend Component'); decomposition.seasonal.plot(ax=axes[1],title='Seasonal Component'); decomposition.resid.plot(ax=axes[2],title='Residual Component'); plt.tight_layout(); plt.savefig(os.path.join(base,'time_series_decomposition.png'),dpi=150); plt.close()
    df['Glucose_MA']=df['Glucose'].rolling(window=7).mean()
    plt.figure(figsize=(12,5)); plt.plot(df['Glucose'],label='Original',alpha=.5); plt.plot(df['Glucose_MA'],label='7-day Moving Average'); plt.legend(); plt.title('Moving Average Smoothing'); plt.tight_layout(); plt.savefig(os.path.join(base,'moving_average.png'),dpi=150); plt.close()
    train_size=int(len(df)*.8); train=df['Glucose'][:train_size]; test=df['Glucose'][train_size:]
    model=ARIMA(train,order=(5,1,0)); fitted=model.fit(); forecast=fitted.forecast(steps=len(test))
    plt.figure(figsize=(12,5)); plt.plot(test.index,test,label='Actual'); plt.plot(test.index,forecast,label='Forecast'); plt.xlabel('Date'); plt.ylabel('Glucose Level'); plt.title('ARIMA Model Forecasting'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(base,'arima_forecast.png'),dpi=150); plt.close()
    print(f'Forecasted {len(test)} future observations using ARIMA(5,1,0).')

if __name__=='__main__':
    run_experiment_16()
