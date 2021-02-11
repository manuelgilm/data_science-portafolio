import numpy as np
from adtk.detector import (ThresholdAD, QuantileAD, InterQuartileRangeAD, 
                            PersistAD, LevelShiftAD, VolatilityShiftAD)


sides = ["negative","positive","both"]
def make_outlier_detector(detector_type, c_parameter, window, side, agg, high, low):

    if detector_type == 'ThresholdAD' or detector_type == 'QuantileAD':
        
        #only high and low parameters are requerided
        if detector_type == 'ThresholdAD':
            return ThresholdAD(low = low, high = high)            
        else:
            return QuantileAD(low = low, high = high)

    if detector_type == 'InterQuartileRangeAD':
        return InterQuartileRangeAD(c_parameter)

    if detector_type == 'PersistAD':
        return PersistAD(c = c_parameter, side = side, window = window)
    
    if detector_type == 'LevelShiftAD':
        return LevelShiftAD(c = c_parameter, side = side, window = window)

    if detector_type == 'VolatilityShiftAD':
        return VolatilityShiftAD(c = c_parameter, side = side, window = window)


def create_clean_time_serie(ts,anomalies):

    anomalies = anomalies.fillna(0)
    anomalies = list(map(bool,anomalies))
    ts.mask(cond = anomalies, other = np.nan,inplace = True)
    return  ts

def validate_high_low_values(detector,high,low):

    if detector == 'ThresholdAD':
        return isinstance(low,int) and isinstance(high,int)
    
    elif detector == 'QuantileAD':
        return high < 1.0 and low > 0.0
    else:
        return True

def validate_parameters(detector,c, window, side, agg, high, low):

    if detector == "ThresholdAD":
        errors = 0 if isinstance(high,int) and isinstance(low, int) and high>low else 1
    
    elif detector == "QuantileAD":
        errors = 0 if high<1 and low >0 else 1
    
    elif detector == "InterQuartileRangeAD":
        errors = 0 if c >0 else 1
    
    elif detector == "PersistAD":
        errors = 0 if c>0 and isinstance(window,int) and side in  sides else 1
    
    elif detector == "LevelShiftAD":
        errors = 0 if c>0 and isinstance(window,int) and side in  sides else 1

    else:
        errors = 0 if c>0 and isinstance(window,int) and side in  sides else 1
        
    return errors