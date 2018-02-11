import pandas as pd
import numpy as np
import datetime
import sqlite3 as sql
from helper_functions import diag_counts

pos = pd.read_pickle("pos.pkl")
neg = pd.read_pickle("neg.pkl")
focus_diags = np.load("focus_diags.npy")

pos_data = diag_counts(pos[pos.diag_cd.isin(focus_diags)])
neg_data = diag_counts(neg[neg.diag_cd.isin(focus_diags)])

pos_data['Target'] = 1
neg_data['Target'] = 0

data = pd.concat([pos_data, neg_data])

data.to_pickle("data.pkl")
