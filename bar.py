import numpy as np
import matplotlib.pyplot as plt
def trend(time, slope=0):
    """A trend over time"""
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    """Adds noise to the series"""
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def plot_series(time, series, format="-", title="", label=None, start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)
def baz():
    print("baz")
def generateTimeSeriesData():
    years=4
    TIME = np.arange(years * 365 + 1, dtype="float32")
    y_intercept = 10
    slope = 0.01
    SERIES = trend(TIME, slope) + y_intercept
    amplitude = 40
    SERIES += seasonality(TIME, period=365, amplitude=amplitude)
    noise_level = 2
    SERIES += noise(TIME, noise_level, seed=42)
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

if __name__ == "__main__": main()