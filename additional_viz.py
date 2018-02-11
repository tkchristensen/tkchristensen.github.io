#additional_viz.py
import pandas as pd
import numpy as np
import sqlite3 as sql
import datetime
import matplotlib.pyplot as plt
from helper_functions import convert_claims_to_diags, diag_counts

conn = sql.connect("heart_disease_claims.db")

focus_diags = np.load("focus_diags.npy")

pos = pd.read_pickle("pos.pkl")
neg = pd.read_pickle("neg.pkl")

pos_data = diag_counts(pos[pos.diag_cd.isin(focus_diags)])
neg_data = diag_counts(neg[neg.diag_cd.isin(focus_diags)])

pos_data['Target'] = 1
neg_data['Target'] = 0

pos_data.Bth_Year.plot(kind='hist', alpha=.3, label='pos', bins=20)
neg_data.iloc[:1100,:].Bth_Year.plot(kind='hist', alpha=.3, label='neg', bins=20)
plt.title("Distribution of Ages in Dataset")
plt.legend()
plt.show()

query = """
select * from pos_claims where
  date(Srvc_Dt) <= date(max_date, '-{0} day') and
  date(Srvc_Dt) >= date(max_date, '-{1} day')
"""

pos_claims30_365 = pd.read_sql_query(query.format(30,365), conn)
pos_claims366_730 = pd.read_sql_query(query.format(366,730), conn)

for t, df in zip(["30-365", "366-730"], [pos_claims30_365, pos_claims366_730]):
    converted_df = convert_claims_to_diags(df)
    converted_df = converted_df.iloc[:,:2].drop_duplicates().groupby('diag_cd').count()
    converted_df.loc[focus_diags].plot(kind='bar')
    plt.title('Counts of Individuals Exhibiting Specific Diagnoses ' + t + ' Days Before Heart Attack')
    plt.show()
