import sqlite3 as sql
import datetime
from matplotlib import pyplot as plt, rcParams
import seaborn
import pandas as pd
import numpy as np
from helper_functions import convert_claims_to_diags

# intialize connection to heart_disease_claims.db database
conn = sql.connect("heart_disease_claims.db")

total_pos_members = conn.execute("select count(distinct sys_id) from pos_claims;").fetchall()
total_pos_claims = conn.execute("select count(*) from pos_claims").fetchall()

query = """
select * from pos_claims where
  date(Srvc_Dt) <= date(max_date, '-{0} day') and
  date(Srvc_Dt) >= date(max_date, '-{1} day')
"""

pos_claims = pd.read_sql_query(query.format(30,730), conn)
pos = convert_claims_to_diags(pos_claims)

pos = pos[~pos.diag_cd.isin(['4274', '4275'])]

pos_diag_counts = pos.iloc[:,:2].drop_duplicates().groupby('diag_cd').count()
sorted_diag_counts = pos_diag_counts.sort_values('sys_id', ascending=False)
sorted_diag_counts.to_pickle("sorted_diag_counts.pkl")

sorted_diag_counts.iloc[:20].plot(kind='bar')
plt.title("Number of Individuals Exhibiting Specific Diagnosis 30-730 Days Prior to Heart Attack")
plt.show()

focus_diags = sorted_diag_counts[:20].index.tolist()
focus_diags = [s.encode('utf8') for s in focus_diags]
focus_diags = np.array(focus_diags)
np.save('focus_diags.npy', focus_diags)

pos = pos[pos.diag_cd.isin(focus_diags)]
pos.to_pickle("pos.pkl")
