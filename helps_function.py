import tensorflow as tf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats.distributions import chi2
from pmdarima.utils import diff, diff_inv


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2-L1))    
    p = chi2.sf(LR, DF).round(3)
    return p
    
def evaluate_preds(y_true, y_pred,):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  mape = tf.keras.losses.MAPE(y_true, y_pred)
  return mape.numpy()

def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))

def inv_diff(off_set, x_diff, differences):
    # Generate np.array for the diff_inv function - it includes first n values(n = 
    # periods) of original data & further diff values of given periods
    x_0 = []
    for i in range(differences, 0, -1):
        if i==1:
            x_diff = np.r_[off_set[0], x_diff].cumsum()
        else:
            x_0 = diff(off_set, 1, i-1)[0]
            x_diff = np.r_[x_0, x_diff].cumsum()
    return x_diff

def evaluate_log_preds(y_true, y_pred):
  y_true = tf.cast(np.exp(y_true), dtype=tf.float32)
  y_pred = tf.cast(np.exp(y_pred), dtype=tf.float32)
  mape = tf.keras.losses.MAPE(y_true, y_pred)

  return mape.numpy()