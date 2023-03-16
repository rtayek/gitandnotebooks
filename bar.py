import numpy as np
import matplotlib.pyplot as plt
def trend(time, slope=0): return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def plot_series(time, series, format="-", title="", label=None, start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)

def plot_series2(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def generateTimeSeriesData(years=4,slope=.01,amplitude=40,period=365,noise_level=2,seed=42):
    TIME = np.arange(years * period + 1, dtype="float32")
    y_intercept = 10
    SERIES = trend(TIME, slope) + y_intercept
    SERIES += seasonality(TIME, period, amplitude=amplitude)
    SERIES += noise(TIME, noise_level, seed)
    return TIME,SERIES

def train_val_split(time, series, time_step):
    # maybe use 70% if time step is none?
    time_train =time[:time_step]
    series_train = series[:time_step]
    time_valid =time[time_step:]
    series_valid = series[time_step:]
    return time_train, series_train, time_valid, series_valid

import tensorflow as tf
from tensorflow import keras as K

def compute_metrics(true_series, forecast):
    mse = K.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = K.metrics.mean_absolute_error(true_series, forecast).numpy()
    return mse, mae

def testComputeMetrics():
    zeros = np.zeros(5)
    ones = np.ones(5)
    mse, mae = compute_metrics(zeros, ones)
    print(f"mse: {mse}, mae: {mae} for series of zeros and prediction of ones\n")
    mse, mae = compute_metrics(ones, ones)
    print(f"mse: {mse}, mae: {mae} for series of ones and prediction of ones\n")
    print(f"metrics are numpy numeric types: {np.issubdtype(type(mse), np.number)}")

def main():
    print('bar main')
    time,series=[],[]
    time,series=generateTimeSeriesData()
    plt.figure(figsize=(10, 6))
    plot_series(time,series)
    print("plot")
    plt.show()

def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
        If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    np_forecast = np.asarray(forecast)
    return np_forecast

if __name__ == "__main__": main()