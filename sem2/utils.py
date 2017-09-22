import os
import numpy as np
import math
import pandas as pd

from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# Quality functions
def qualitySSE(x,y):
    # Sum squared error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return ((x-y)**2).sum(), (x-y)**2

def qualityMSE(x,y):
    # Mean squared error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return ((x-y)**2).mean() , (x-y)**2

def qualityRMSE(x,y):
    # Root mean squared error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return (((x-y)**2).mean())**(0.5) , (x-y)**2

def qualityMAE(x,y):
    # Mean absolute error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return (x-y).abs().mean(), (x-y).abs()

def qualityMAPE(x,y):
    # Mean absolute percentage error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    qlt = ((x-y).abs()/x).replace([np.inf, -np.inf], np.nan)
    return qlt.mean() , (x-y).abs()

def qualityMACAPE(x,y):
    # Mean average corrected absolute percentage error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    qlt = (2*(x-y).abs()/(x+y)).replace([np.inf, -np.inf], np.nan)
    return qlt.mean() , (x-y).abs()

def qualityMedianAE(x,y):
    # Median absolute error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return ((x-y).abs()).median(), (x-y).abs()

def BuildForecast(h, ts, AlgName, AlgTitle, ParamsArray, step='D'):
	FRC_TS = dict()
	for p in ParamsArray:
		frc_horizon = pd.date_range(ts.index[-1], periods=h+1, freq=step)[1:]
		frc_ts = pd.DataFrame(index = ts.index.append(frc_horizon), columns = ts.columns)
		
		for cntr in ts.columns:
			frc_ts[cntr] = eval(AlgName)(ts[cntr], h, p)
		
#         frc_ts.columns = frc_ts.columns+('%s %s' % (AlgTitle, p))
		FRC_TS['%s %s' % (AlgTitle, p)] = frc_ts
	return FRC_TS

def plotTSForecast(ts, frc_ts, ts_num=0, alg_title=''):
	frc_ts.columns = ts.columns+'; '+alg_title
	ts[ts.columns[0]].plot(style='b', linewidth=1.0, marker='o')
	ax = frc_ts[frc_ts.columns[0]].plot(style='r-^', figsize=(15,5), linewidth=1.0)
	plt.xlabel('Time ticks')
	plt.ylabel('TS values')
	plt.legend()
	return ax
	
def InitExponentialSmoothing(x, h, Params):

    T = len(x)
    alpha = Params['alpha']
    AdaptationPeriod=Params['AdaptationPeriod']
    FORECAST = [np.NaN]*(T+h)
    if alpha>1:
        w.warn('Alpha can not be more than 1')
        #alpha = 1
        return FORECAST
    if alpha<0:
        w.warn('Alpha can not be less than 0')
        #alpha = 0
        return FORECAST
    y = x[0]
    t0=0
    for t in range(0, T):
        if not math.isnan(x[t]):
            if math.isnan(y):
                y=x[t]
                t0=t
            if (t-t0+1)<AdaptationPeriod:
                y = y*(1-alpha*(t-t0+1)/(AdaptationPeriod)) + alpha*(t-t0+1)/(AdaptationPeriod)*x[t]
            y = y*(1-alpha) + alpha*x[t]
            #else do not nothing
        FORECAST[t+h] = y
    return FORECAST	

# Start with this code
###################### Winters Exponential Smoothing #########################
# x <array Tx1>- time series, 
# h <scalar> - forecasting delay
# Params <dict> - dictionary with 
#    alpha <scalar in [0,1]> - smoothing parameter
#    delts <scalar in [0,1]> - seasonality smoothing parameter

def WintersExponentialSmoothing(x, h, Params):
    T = len(x)
    alpha = Params['alpha']
    gamma = Params['gamma']
    p = Params['seasonality_period']
    
    FORECAST = [np.NaN] * (T + h)
    
    s = [np.NaN] * T
    
    # set cool init values for 1st season from wiki (triple exp. smoothing)
    N = int(T / p)
    A = np.zeros(N)
    for j in range(N):
        A[j] = np.sum([x[p*j + i] for i in range(p)])
    A /= p    
    
    for t in range(p):
        s[t] = (np.array(x[t:p*N:p]) / A).sum() / N
    
    l = x[p]
    for t in range(p, T):
        if not math.isnan(x[t]):           
            l_prev = l 
            l =  alpha * (x[t] - s[t-p]) + (1 - alpha) * l
            s[t] = gamma * (x[t] - l_prev) + (1 - gamma) * s[t-p]
            
        FORECAST[t + h] = l + s[t - p + h%p]
    return FORECAST

def MultiWintersExponentialSmoothing(x, h, Params):
    T = len(x)
    alpha = Params['alpha']
    gamma = Params['gamma']
    p = Params['seasonality_period']
    
    FORECAST = [np.NaN] * (T + h)
    
    s = [np.NaN] * T
    
    # set cool init values for 1st season from wiki (triple exp. smoothing)
    N = int(T / p)
    A = np.zeros(N)
    for j in range(N):
        A[j] = np.sum([x[p*j + i] for i in range(p)])
    A /= p    
    
    for t in range(p):
        s[t] = (np.array(x[t:p*N:p]) / A).sum() / N
    
    l = x[p]
    for t in range(p, T):
        if not math.isnan(x[t]):           
            l_prev = l 
            l =  alpha * (x[t] / s[t-p]) + (1 - alpha) * l
            s[t] = gamma * (x[t] / l_prev) + (1 - gamma) * s[t-p]
            
        FORECAST[t + h] = l * s[t - p + h%p]
    return FORECAST


def TheilWageExponentialSmoothing(x, h, Params):
    T = len(x)
    alpha = Params['alpha']
    beta = Params['beta']
    gamma = Params['gamma']
    p = Params['seasonality_period']
    
    FORECAST = [np.NaN] * (T + h)
    
    s = [np.NaN] * T
    b = (np.sum(x[p:2*p]) - np.sum(x[:p])) / p ** 2
    
    N = int(T / p)
    A = np.zeros(N)
    for j in range(N):
        A[j] = np.sum([x[p*j + i] for i in range(p)])
    A /= p    
    
    for t in range(p):
        s[t] = (np.array(x[t:p*N:p]) / A).sum() / N
    
    l = x[p]
    for t in range(p, T):
        if not math.isnan(x[t]):           
            l_prev = l 
            l = alpha * (x[t] - s[t-p]) + (1 - alpha) * (l + b)
            b = beta * (l - l_prev) + (1 - beta) * b
            s[t] = gamma * (x[t] - l) + (1 - gamma) * s[t-p]
            
        FORECAST[t + h] = l + h * b + s[t - p + h%p]
    return FORECAST
