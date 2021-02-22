import numpy as np
from adtk.detector import ThresholdAD


sides = ["negative","positive","both"]
def make_outlier_detector(detector_type, high,low):
  
    return ThresholdAD(low = low, high = high)

def create_clean_time_serie(ts,anomalies):

    anomalies = anomalies.fillna(0)
    anomalies = list(map(bool,anomalies))
    ts.mask(cond = anomalies, other = np.nan,inplace = True)
    return  ts

def validate_high_low_values(detector,high,low):
    
    return isinstance(low,int) and isinstance(high,int)














    







