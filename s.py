class S(): # class for series
    def __init__(self,series=None):
        self.series=series
    def __str__(self):
        if self.series is not None: return "series: "+str(type(self.series))
        else: return "series: "+str(None)
    def trend(time, slope=0):
        return slope * time
    
    def seasonal_pattern(season_time,periods=3):
        """An arbitrary pattern"""
        return np.where(season_time < 0.1,
                        np.cos(season_time * periods*2 * np.pi), 
                        2 / np.exp(9 * season_time))
    
    def seasonality(time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * seasonal_pattern(season_time)
    
    def noise(time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level        
    #TIME, SERIES = generate_time_series()
    #splitTie = 1100
    #windoeSize = 20
    #batchSize = 32
    #shuffkeBufferSize = 1000
def main():
  print('main')
  s=S()
  print(s)    

if __name__ == "__main__": main()
