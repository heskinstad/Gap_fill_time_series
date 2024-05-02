import numpy as np
import scipy.stats as st

def mean_confidence_interval(data, confidence_value=0.95):
    return st.t.interval(confidence_value, len(data)-1, loc=np.mean(data), scale=st.sem(data))
