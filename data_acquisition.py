from localLibrary_DatabaseSchema import *
from localLibrary_dataQueries import *
import pandas as pd

STM_queries = {'SG_STM':SG_STM,'MK_STM':MK_STM, 'CRM_STM':CRM_STM, 'STM_TM':STM_TM, 'STM_purchase':STM_purchase}
                #'YZ_STM':YZ_STM,'FTS_STM':FTS_STM}
nonSTM_queries = {'SG_nonSTM':SG_nonSTM,'MK_nonSTM':MK_nonSTM, 'CRM_nonSTM':CRM_nonSTM}
                #'YZ_nonSTM':YZ_nonSTM, 'FTS_nonSTM':FTS_nonSTM,

lost_queries = {'SG_lost':SG_lost,'MK_lost':MK_lost, 'CRM_lost':CRM_lost}
                #'YZ_lost':YZ_lost,'FTS_lost':FTS_lost,

for query in STM_queries:
    cursor.execute(STM_queries[query])
    df_columns = [column[0] for column in cursor.description]
    df_data = cursor.fetchall()
    df_arr = []
    for row in df_data:
        df_arr.append(dict(zip(df_columns, row)))
    df = pd.DataFrame(df_arr, columns = df_columns)
    df.to_csv('data/' + query + '.csv', index = False)

for query in nonSTM_queries:
    cursor.execute(nonSTM_queries[query])
    df_columns = [column[0] for column in cursor.description]
    df_data = cursor.fetchall()
    df_arr = []
    for row in df_data:
        df_arr.append(dict(zip(df_columns, row)))
    df = pd.DataFrame(df_arr, columns = df_columns)
    df.to_csv('data/' + query + '.csv', index = False)

for query in lost_queries:
    cursor.execute(lost_queries[query])
    df_columns = [column[0] for column in cursor.description]
    df_data = cursor.fetchall()
    df_arr = []
    for row in df_data:
        df_arr.append(dict(zip(df_columns, row)))
    df = pd.DataFrame(df_arr, columns = df_columns)
    df.to_csv('data/' + query + '.csv', index = False)
