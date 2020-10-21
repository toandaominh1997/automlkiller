import os
import numpy as np
import pandas as pd
from IPython.display import display, HTML, clear_output, update_display
import pandas_profiling

class Dataset(object):
    def __init__(self):
        super(Dataset, self).__init__()

    @classmethod
    def profile_data(cls, data, profile = True, verbose = True):
        if os.path.isfile(data):
            dataset = pd.read_csv(data)
        else:
            dataset = data
        if profile:
            pf = pandas_profiling.ProfileReport(dataset.copy())
            display(pf)
        return dataset
