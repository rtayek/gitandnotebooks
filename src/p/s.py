import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt


class S():  # class for series

    def __init__(self, time=None, series=None):
        self.time = time
        self.series = series

    def __str__(self):
        if self.time is not None: return "time: " + str(type(self.time))
        if self.series is not None: return "series: " + str(type(self.series))
        else: return "series: " + str(None)

    def trend(self, time, slope=0):
        return slope * time
    
    def seasonal_pattern(self, season_time, periods=3):
        """An arbitrary pattern"""
        return np.where(season_time < 0.1,
                        np.cos(season_time * periods * 2 * np.pi),
                        2 / np.exp(9 * season_time))
    
    def seasonality(self, time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * self.seasonal_pattern(season_time)
    
    def noise(self, time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level   

    def generate_time_series(self):
        time = np.arange(4 * 365 + 1, dtype="float32")
        y_intercept = 10
        slope = 0.005
        series = self.trend(time, slope) + y_intercept
        amplitude = 50
        series += self.seasonality(time, period=365, amplitude=amplitude)
        noise_level = 3
        series += self.noise(time, noise_level, seed=51)
        return time, series
    
    @staticmethod
    def train_val_split(self, time, series, time_step):
        time_train = time[:time_step]
        series_train = series[:time_step]
        time_valid = time[time_step:]
        series_valid = series[time_step:]
        return time_train, series_train, time_valid, series_valid

    @staticmethod
    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(False)     


class G():

    def __init__(self):
        self.s = S()
        TIME, SERIES = self.s.generate_time_series()
        self.TIME=TIME
        self.SERIES=SERIES
        self.SPLIT_TIME = 1100
        self.WINDOW_SIZE = 20
        self.BATCH_SIZE = 32
        self.SHUFFLE_BUFFER_SIZE = 1000    


def main():
    print('main')
    s=S()
    g=G()
    x=g.TIME
    y=g.SERIES
    plt.figure(figsize=(10, 6))
    S.plot_series(x,y)
    plt.show()


if __name__ == "__main__": main()
