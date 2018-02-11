import pandas as pd
import numpy as np
import sqlite3 as sql
import datetime

def convert_claims_to_diags(df):
    """
    Takes a line of claims and translates each diagnosis into its own row.
    """
    data = []
    for r in df.values[:,:-1]:
        for i in [1,2,3]:
            data.append([r[0], r[i], r[4]])

    out = pd.DataFrame(data, columns=['sys_id', 'diag_cd', 'date'])
    out = out[~out.diag_cd.isin(['UNK',''])]
    out.sys_id = out.sys_id.astype(int)
    out.date = out.date.apply(lambda s : datetime.datetime.strptime(s, "%Y-%m-%d"))
    return out

def diag_counts(dx):
    """
    Inputs:
        dx - DataFrame of diagnoses with columns [ID, Diag_Cd, Date]

    Returns:
        data - DataFrame of diagnosis counts

    """
    metro = sql.connect("../../../../metro.db")

    unique_id = dx.sys_id.unique()
    unique_id = unique_id[np.argsort(unique_id)]

    id2index = {j:i for i,j in zip(np.arange(unique_id.size), unique_id)}

    unique_diags = dx.diag_cd.unique()
    unique_diags = unique_diags[np.argsort(unique_diags)]
    diag2index = {j:i for i,j in zip(np.arange(unique_diags.size), unique_diags)}

    co_occur = np.zeros((unique_id.size, unique_diags.size)).astype(int)
    for id_, diag in dx.iloc[:,:2].values:
        i = id2index[id_]
        j = diag2index[diag]
        co_occur[i,j] += 1

    data = pd.DataFrame(co_occur, columns=unique_diags)
    data["ID"] = unique_id
    birth_dates = pd.read_sql_query("""select DI_Indv_Sys_Id as ID, date(Bth_Dt) as Bth_Dt from member
                                    where DI_Indv_Sys_Id in {0}""".format(tuple(unique_id)), metro)
    data = pd.merge(data, birth_dates, on='ID').copy()
    data.Bth_Dt = data.Bth_Dt.apply(lambda s : datetime.datetime.strptime(s, '%Y-%m-%d'))

    data["Bth_Year"] = data.Bth_Dt.apply(lambda s : s.year)
    data.index = data['ID']
    data.drop(["ID", "Bth_Dt"], axis=1, inplace=True)

    return data
