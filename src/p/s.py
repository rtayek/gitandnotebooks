import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as K
print(tf.__version__)

# copy from week 1 notebook


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

def week1():
    TIME = np.arange(4 * 365 + 1, dtype="float32")
    y_intercept = 10
    slope = 0.01
    SERIES = trend(TIME, slope) + y_intercept
    amplitude = 40
    SERIES += seasonality(TIME, period=365, amplitude=amplitude)
    # Adding some noise
    noise_level = 2
    SERIES += noise(TIME, noise_level, seed=42)
    return TIME, SERIES


class S():  # class for series

    def __init__(self, time=None, series=None):
        self.time = time
        self.series = series

    @classmethod
    def create(clazz):
        clazz  # shut up lint
        s = S()
        s.time, s.series = s.generate_time_series()
        return s
        
    SPLIT_TIME = 1100
    WINDOW_SIZE = 20
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000    

    def __str__(self):
        if self.time is not None: return "time: " + str(type(self.time))
        if self.series is not None: return "series: " + str(type(self.series))
        else: return "series: " + str(None)

    def trend(self, time, slope=0):
        return slope * time
    
    def seasonal_pattern(self, season_time, periods=3):  # halfPeriods?
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

    def generateTimeSeriesDataForWeek1(self, years=4, slope=.01, amplitude=40, period=365, noise_level=2, seed=42):
        print("i am here")
        TIME = np.arange(years * 365 + 1, dtype="float32")
        
        y_intercept = 10
        SERIES = self.trend(TIME, slope=slope) + y_intercept
        SERIES += self.seasonality(TIME, period=period, amplitude=amplitude)
        SERIES += self.noise(TIME, noise_level, seed)
        return TIME, SERIES

    def generate_time_series(self):  # maybe for week 2?
        time = np.arange(4 * 365 + 1, dtype="float32")
        y_intercept = 10
        slope = 0.005
        series = self.trend(time, slope) + y_intercept
        amplitude = 50
        series += self.seasonality(time, period=365, amplitude=amplitude)
        noise_level = 3
        series += self.noise(time, noise_level, seed=51)
        return time, series
    
    # week 1
    def generateTimeSeriesData(self, years=4, slope=.01, amplitude=40, period=365, noise_level=2, seed=42):
        TIME = np.arange(years * period + 1, dtype="float32")
        y_intercept = 10
        SERIES = self.trend(TIME, slope) + y_intercept
        SERIES += self.seasonality(TIME, period, amplitude=amplitude)
        SERIES += self.noise(TIME, noise_level, seed)
        return TIME, SERIES

    def split(self, val, test=None):
        if test is  None: test = len(self.series)
        t = self.time[0:val]
        s = self.series[0:val]
        tV = self.time[val:test]                                                                
        sV = self.series[val:test]
        tT = self.time[test: len(self.series)]
        sT = self.series[test: len(self.series)]
        print(test)
        return S(t, s), S(tV, sV), S(tT, sT)

    def compute_metrics(self,other):
        mse = K.metrics.mean_squared_error(self.series, other.series).numpy()
        mae = K.metrics.mean_absolute_error(self.series,other.series).numpy()
        return mse, mae
        
    @staticmethod
    def static_compute_metrics(clazz,true_series, forecast):
        mse = K.metrics.mean_squared_error(true_series, forecast).numpy()
        mae = K.metrics.mean_absolute_error(true_series, forecast).numpy()
        return mse, mae

    @staticmethod
    def train_val_split(clazz, time, series, time_step):
        # maybe use 70% if time step is none?
        clazz  # shut up lint
        time_train = time[:time_step]
        series_train = series[:time_step]
        time_valid = time[time_step:]
        series_valid = series[time_step:]
        return time_train, series_train, time_valid, series_valid

    def plot_series(self, tamrof="-", title="", label=None, start=0, end=None):
        plt.plot(self.time[start:end], self.series[start:end], tamrof, label=label)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(title)
        if label:
            plt.legend()
        plt.grid(False)  # was True

    def plot0(self, title=""):
        plt.figure(figsize=(10, 6))
        self.plot_series(title=title)
        plt.show()   

    @staticmethod
    @tf.autograph.experimental.do_not_convert  # maybe i can remove this?
    def windowed_dataset(series, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER_SIZE):
        print("series in windowed dataset", type(series), series.shape)
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
        print("window size:", window_size)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    def checkWeek1Series(self, n=5):
        print("all:", self.time, self.series)
        print("time: " + str(self.time[:n]), str(self.time[-n:]))
        print("series: " + str(self.series[:n]), str(self.series[-n:]))
        exTStart = [0., 1., 2., 3., 4.] 
        exTStart = np.array(exTStart, dtype=np.float32)
        exTEnd = [1456., 1457., 1458., 1459., 1460.]
        exTEnd = np.array(exTEnd, dtype=np.float32)
        exSStart = [50.993427, 49.680145, 51.102207, 52.596966, 48.7213  ]
        exSStart = np.array(exSStart, dtype=np.float32)
        exSEnd = [26.07758, 25.342949 , 27.169878 , 25.946482 , 64.32309 ]
        exSEnd = np.array(exSEnd, dtype=np.float32)
        print("expected:", type(exTStart))
        print(type(self.time), type(self.time[0]))
        print("expected:", type(exSStart))
        print(type(self.series), type(self.series[0]))
        print("time:", self.time[:n])
        print("expected", exTStart)
        t1 = np.allclose(exTStart, self.time[:n])
        print("diff 1 =", exTStart - self.time[:n])
        t2 = np.allclose(exTEnd, self.time[-n:])
        print("diff 2", exTEnd - self.time[-n:])
        t3 = np.allclose(exSStart, self.series[:n])
        print("diff 3", exSStart - self.series[:n])
        t4 = np.allclose(exSEnd, self.series[-n:])
        print("diff 4", exSEnd - self.series[-n:])
        print(t1, t2, t3, t4)
        if not (self.time[:n] - exTStart).any():
            print("badness 1")
            print((self.time[:n] - exTStart).any())
        print("---")        
        print(self.time[-n:])
        print(exTEnd)
        if not (self.time[-n:] - exTEnd).any(): print("badness 2")
        print("---")        
        print(self.series[:n])
        print(exSStart)
        if not (self.series[:n] - exSStart).any(): print("badness 3")
        print("---")        
        print(self.series[-n:])
        print(exSEnd)
        if not (self.series[-n:] - exSEnd).any(): print("badness 4")


def windowed_dataset(series, window_size=S.WINDOW_SIZE, batch_size=S.BATCH_SIZE, shuffle_buffer=S.SHUFFLE_BUFFER_SIZE):
    return S.windowed_dataset(series, window_size, batch_size, shuffle_buffer)


def testWindoedSDataset():
    s = S.create()
    tT, sT, tV, sV, _, _ = s.split(val=S.SPLIT_TIME)
    test_dataset = S.windowed_dataset(sT, window_size=1, batch_size=5, shuffle_buffer=1)
    batch_of_features, batch_of_labels = next((iter(test_dataset)))
    tT, sT, tV, sV  # shut up lint
    print(f"batch_of_features has type: {type(batch_of_features)}\n")
    print(f"batch_of_labels has type: {type(batch_of_labels)}\n")
    print(f"batch_of_features has shape: {batch_of_features.shape}\n")
    print(f"batch_of_labels has shape: {batch_of_labels.shape}\n")
    print(f"batch_of_features is equal to first five elements in the series: {np.allclose(batch_of_features.numpy().flatten(), sT[:5])}\n")
    print(f"batch_of_labels is equal to first five labels: {np.allclose(batch_of_labels.numpy(), sT[1:6])}")


def testComputeMetrics():
    zeros = np.zeros(5)
    ones = np.ones(5)
    s=S(None,zeros)
    other=S(None,ones)
    
    
#    mse, mae = S.compute_metrics(zeros, ones)
    mse, mae = s.compute_metrics(other)
    print(f"mse: {mse}, mae: {mae} for series of zeros and prediction of ones\n")
#    mse, mae = S.compute_metrics(ones, ones)
    mse, mae = other.compute_metrics(other)
    print(f"mse: {mse}, mae: {mae} for series of ones and prediction of ones\n")
    print(f"metrics are numpy numeric types: {np.issubdtype(type(mse), np.number)}")


def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
        If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    np_forecast = np.asarray(forecast)
    return np_forecast


def main():
    print('main')
    testComputeMetrics()
    s = S.create()
    print(s)
    # plot(g.TIME,g.SERIES)
#    (x,y,z,w) = S.train_val_split(g.TIME, g.SERIES,time_step=g.SPLIT_TIME)
#    print(S.SHUFFLE_BUFFER_SIZE)


if __name__ == "__main__": main()
