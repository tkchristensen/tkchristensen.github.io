import numpy as np
import pandas as pd
import sqlite3 as sql
import datetime
import seaborn
import matplotlib.pyplot as plt
from helper_functions import convert_claims_to_diags

sorted_diag_counts = pd.read_pickle("sorted_diag_counts.pkl")
focus_diags = np.load('focus_diags.npy')

conn = sql.connect("heart_disease_claims.db")

neg_members = pd.read_sql_query("select sys_id from neg_members", conn)
neg_members = neg_members.values.flatten()

subset_negmem = neg_members[np.random.randint(0, len(neg_members), 1100)]

query = """
select * from neg_claims
where sys_id in {0}
""".format(tuple(subset_negmem))

neg_claims = pd.read_sql_query(query, conn)
neg = convert_claims_to_diags(neg_claims)
neg = neg[neg.diag_cd.isin(focus_diags)]
neg.to_pickle("neg.pkl")

neg_diag_counts = neg.iloc[:,:2].drop_duplicates().groupby('diag_cd').count()

sorted_neg_diag_counts = neg_diag_counts.loc[focus_diags]

counts = np.zeros((20,2))
counts[:,0] = sorted_diag_counts.iloc[:20].values.flatten()
counts[:,1] = sorted_neg_diag_counts.values.flatten()


sorted_neg_diag_counts = pd.DataFrame(counts,
                                      columns=['pos', 'neg'],
                                      index=focus_diags)

sorted_neg_diag_counts.plot(kind='bar')
plt.title("Number of Individuals Exhibiting Specific Diagnosis Within 2 Years")
plt.show()
