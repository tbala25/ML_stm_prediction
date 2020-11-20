################################################################################
################################### IMPORTS ####################################
################################################################################
from localLibrary_AWSConnector import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold
import sklearn.preprocessing as sk


################################################################################
############################## HELPER FUNCTIONS ################################
################################################################################
#Returns dictionary of dataframes
def get_data():
    data_list = []
    data_dict = {}

    file_dir = os.getcwd() + '\\data\\'
    items = os.listdir(file_dir)

    for i in items:
        if i == 'SG_STM_purchase_date.csv' or i == 'STM_TM.csv':
            pass
        else:
            file = i.replace('.csv', '')
            data_list.append(file)

            # Save STM files as DataFrames
            data_dict[file] = pd.read_csv(file_dir + i, index_col=0)

    print(data_list)

    return data_dict

#Convert time helper
def _convert_time_to_int(time):
    #print(time)
    if time != time:
        return 0
    elif time == 0:
        return 0
    else:
        return int(''.join(c for c in time[:10] if c.isdigit()))

#convert date columns to int  for given df
def convert_time_int(df):

    df['EarliestCRM_int'] = [_convert_time_to_int(x) for x in df['EarliestCRM']]
    df['LatestCRM_int'] = [_convert_time_to_int(x) for x in df['LatestCRM']]

    df['LatestSeatGeek_int'] = [_convert_time_to_int(x) for x in df['LatestSeatGeekDate']]
    df['EarliestSeatGeek_int'] = [_convert_time_to_int(x) for x in df['EarliestSeatGeekDate']]

    df['EarliestMarketo_int'] = [_convert_time_to_int(x) for x in df['EarliestMarketoDate']]
    df['LatestMarketo_int'] = [_convert_time_to_int(x) for x in df['LatestMarketoDate']]

    df['EarliestFanatics_int'] = [_convert_time_to_int(x) for x in df['EarliestFanaticsDate']]
    df['LatestFanatics_int'] = [_convert_time_to_int(x) for x in df['LatestFanaticsDate']]

    df['EarliestYinzcam_int'] = [_convert_time_to_int(x) for x in df['EarliestYinzcamDate']]
    df['LatestYinzcam_int'] = [_convert_time_to_int(x) for x in df['LatestYinzcamDate']]

    df['Purchase Date'] = [_convert_time_to_int(x) for x in df['Purchase Date']]

    return df

#Creates date difference column (latest - earliest)
def calculate_time_diff(df):

    df['CRM_diff'] = df['LatestCRM_int'] - df['EarliestCRM_int']
    df['SeatGeek_diff'] = df['LatestSeatGeek_int'] - df['EarliestSeatGeek_int']
    df['Marketo_diff'] = df['LatestMarketo_int'] - df['EarliestMarketo_int']
    df['Fanatics_diff'] = df['LatestFanatics_int'] - df['EarliestFanatics_int']
    df['Yinzcam_diff'] = df['LatestYinzcam_int'] - df['EarliestYinzcam_int']


    df['SeatGeek_to_purchase'] = [p-s if s > 0 else p for s,p in zip(df['LatestSeatGeek_int'], df['Purchase Date'])]
    df['CRM_to_purchase'] = [p-s if s > 0 else p for s,p in zip(df['LatestCRM_int'], df['Purchase Date'])]
    df['Marketo_to_purchase'] = [p-s if s > 0 else p for s,p in zip(df['LatestMarketo_int'], df['Purchase Date'])]
    df['Fanatics_to_purchase'] = [p-s if s > 0 else p for s,p in zip(df['LatestFanatics_int'], df['Purchase Date'])]
    df['Yinzcam_to_purchase'] = [p-s if s > 0 else p for s,p in zip(df['LatestYinzcam_int'], df['Purchase Date'])]

#     df['SeatGeek_to_purchase'] = [p-s if p > 0 else 0 for s,p in zip(df['LatestSeatGeek_int'], df['Purchase Date'])]
#     df['CRM_to_purchase'] = [p-s if p > 0 else 0 for s,p in zip(df['LatestCRM_int'], df['Purchase Date'])]
#     df['Marketo_to_purchase'] = [p-s if p > 0 else 0 for s,p in zip(df['LatestMarketo_int'], df['Purchase Date'])]
#     df['Fanatics_to_purchase'] = [p-s if p > 0 else 0 for s,p in zip(df['LatestFanatics_int'], df['Purchase Date'])]
#     df['Yinzcam_to_purchase'] = [p-s if p > 0 else 0 for s,p in zip(df['LatestYinzcam_int'], df['Purchase Date'])]

    return df

################################################################################
##################### DATA CLEANING / FEATURE ENGINEERING ######################
################################################################################

################################################################################
## GET DATA
all_data = get_data()

################################################################################
## PIVOT SG & MARKETO DATA

for key in all_data.keys():
    #SEATGEEK PIVOT
    if 'SG' in key:
        pivoted = pd.pivot_table(all_data[key],
                         values=['TotalSeatGeekTransactions','TotalTicketVolume','TotalScannedTicketVolume','TotalTicketDollarValue'],
                         index=['SSB_CRMSYSTEM_CONTACT_ID'],
                         columns=['cjsgActivityType', 'cjsgSecondaryTicketType'],
                         aggfunc=(np.sum),
                         fill_value=0)
        #pivoted.columns = [' '.join(col).strip() for col in pivoted.columns.values]
        pivoted = pd.DataFrame(pivoted.to_records())

        pivoted_agg = pd.DataFrame()
        pivoted_agg['SSB_CRMSYSTEM_CONTACT_ID'] = pivoted['SSB_CRMSYSTEM_CONTACT_ID']
        pivoted_agg['total_scanned'] = pivoted["('TotalScannedTicketVolume', 'Purchase', 'Primary')"] + pivoted["('TotalScannedTicketVolume', 'Purchase', 'Resale')"] + pivoted["('TotalScannedTicketVolume', 'Purchase', 'Transfer')"]

        pivoted_agg['primary_purchase_transactions'] = pivoted["('TotalSeatGeekTransactions', 'Purchase', 'Primary')"]
        pivoted_agg['secondary_purchase_transactions'] = pivoted["('TotalSeatGeekTransactions', 'Purchase', 'Resale')"] + pivoted["('TotalSeatGeekTransactions', 'Purchase', 'Transfer')"]
        pivoted_agg['secondary_sell_transactions'] = pivoted["('TotalSeatGeekTransactions', 'Sell', 'Resale')"] + pivoted["('TotalSeatGeekTransactions', 'Sell', 'Transfer')"]

        pivoted_agg['primary_purchase_dollars'] = pivoted["('TotalTicketDollarValue', 'Purchase', 'Primary')"]
        pivoted_agg['secondary_purchase_dollars'] = pivoted["('TotalTicketDollarValue', 'Purchase', 'Resale')"] + pivoted["('TotalTicketDollarValue', 'Purchase', 'Transfer')"]
        pivoted_agg['secondary_sell_dollars'] = pivoted["('TotalTicketDollarValue', 'Sell', 'Resale')"] + pivoted["('TotalTicketDollarValue', 'Sell', 'Transfer')"]

        pivoted_agg['primary_purchase_tickets'] = pivoted["('TotalTicketVolume', 'Purchase', 'Primary')"]
        pivoted_agg['secondary_purchase_tickets'] = pivoted["('TotalTicketVolume', 'Purchase', 'Resale')"] + pivoted["('TotalTicketVolume', 'Purchase', 'Transfer')"]
        pivoted_agg['secondary_sell_tickets'] = pivoted["('TotalTicketVolume', 'Sell', 'Resale')"] + pivoted["('TotalTicketVolume', 'Sell', 'Transfer')"]

        min_dates = all_data[key]['EarliestSeatGeekDate'].groupby(['SSB_CRMSYSTEM_CONTACT_ID']).min()
        max_dates = all_data[key]['LatestSeatGeekDate'].groupby(['SSB_CRMSYSTEM_CONTACT_ID']).max()
        pivoted_agg = pivoted_agg.merge(min_dates, on = 'SSB_CRMSYSTEM_CONTACT_ID')
        pivoted_agg = pivoted_agg.merge(max_dates, on = 'SSB_CRMSYSTEM_CONTACT_ID')

        all_data[key] = pivoted_agg

    #MARKETO PIVOT
    if 'MK' in key:
        pivoted = pd.pivot_table(all_data[key],
                         values=['TotalMarketoVolume'],
                         index=['SSB_CRMSYSTEM_CONTACT_ID'],
                         columns=['cjmktActivityType'],
                         aggfunc=(np.sum),
                         fill_value=0)

        pivoted.columns = pivoted.columns.droplevel(0)
        min_dates = all_data[key]['EarliestMarketoDate'].groupby(['SSB_CRMSYSTEM_CONTACT_ID']).min()
        max_dates = all_data[key]['LatestMarketoDate'].groupby(['SSB_CRMSYSTEM_CONTACT_ID']).max()
        pivoted = pivoted.merge(min_dates, on = 'SSB_CRMSYSTEM_CONTACT_ID')
        pivoted = pivoted.merge(max_dates, on = 'SSB_CRMSYSTEM_CONTACT_ID')

        all_data[key] = pivoted

################################################################################
## MERGE ALL DF
STM = None
nonSTM = None
lost = None
for key in all_data.keys():
    df = all_data[key]
    if('STM' in key and 'non' not in key):
        if STM is None:
            STM = df
        else:
            STM = STM.merge(df, how = 'outer', on = 'SSB_CRMSYSTEM_CONTACT_ID')

    elif('non' in key):
        if nonSTM is None:
            nonSTM = df
        else:
            nonSTM = nonSTM.merge(df, how = 'outer', on = 'SSB_CRMSYSTEM_CONTACT_ID')
    elif('lost' in key):
        if lost is None:
            lost = df
        else:
            lost = lost.merge(df, how = 'outer', on = 'SSB_CRMSYSTEM_CONTACT_ID')

################################################################################
## DATE FEATURES ENGINEERING

#CONVERT DATE COLUMNS TO INT
nonSTM['Purchase Date'] = datetime.datetime.now().strftime('%Y-%m-%d')
lost['Purchase Date'] = datetime.datetime.now().strftime('%Y-%m-%d')

STM = convert_time_int(STM)
nonSTM = convert_time_int(nonSTM)
lost = convert_time_int(lost)

#CALCULATE DATE DIFFERENCE
#QUANTIFY LENGTH OF ENGAGEMENT
STM = calculate_time_diff(STM)
nonSTM = calculate_time_diff(nonSTM)
lost = calculate_time_diff(lost)

#DROP DATE COLUMNS
for col in STM.columns:
    if 'Date' in col:
        STM.drop([col], axis=1, inplace = True)
        nonSTM.drop([col], axis=1, inplace = True)
        lost.drop([col], axis=1, inplace = True)
    elif '_int' in col:
        print(col)
        STM.drop([col], axis=1, inplace = True)
        nonSTM.drop([col], axis=1, inplace = True)
        lost.drop([col], axis=1, inplace = True)

    else:
        pass

STM.drop(['EarliestCRM', 'LatestCRM'], axis=1, inplace = True)
nonSTM.drop(['EarliestCRM', 'LatestCRM'], axis=1, inplace = True)
lost.drop(['EarliestCRM', 'LatestCRM'], axis=1, inplace = True)

################################################################################
##
#Fill NA
STM = STM.fillna(0)
nonSTM = nonSTM.fillna(0)
lost = lost.fillna(0)

#Drop Old Features
for c in lost.columns.values:
    if 'zz' in c:
        print(c)
        lost.drop([c], axis = 0, inplace = True)
    if 'Group Form' in c:
        lost.drop([c], axis = 1, inplace = True)

################################################################################
## EXCLUDE RENEWALS
stm_purchase = pd.read_csv('./data/STM_TM.csv')
STM = pd.merge(STM, stm_purchase, left_on='SSB_CRMSYSTEM_CONTACT_ID',right_on = 'SSB_Composite_Record_Unique_ID', how="outer", indicator=True
              ).query('_merge=="left_only"')
nonSTM = pd.merge(nonSTM, stm_purchase, left_on='SSB_CRMSYSTEM_CONTACT_ID',right_on = 'SSB_Composite_Record_Unique_ID', how="outer", indicator=True
              ).query('_merge=="left_only"')
lost = pd.merge(lost, stm_purchase, left_on='SSB_CRMSYSTEM_CONTACT_ID',right_on = 'SSB_Composite_Record_Unique_ID', how="outer", indicator=True
              ).query('_merge=="left_only"')

################################################################################
## STACK STM NONSTM, LOST

STM['target'] = '1'
lost['target'] = '0'
nonSTM['target'] = 'nonSTM'

full_data = pd.concat([STM, lost])
full_data.drop(['acct_id', 'SSB_Composite_Record_Unique_ID', '_merge' ], axis = 1, inplace = True)

################################################################################
############################### MODEL BUILDING #################################
################################################################################

def variance_threshold_selector(data, threshold=0.5):
    # https://stackoverflow.com/a/39813304/1956309
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

# min_variance = .9 * (1 - .9)  # You can play here with different values.
min_variance = 3
df = full_data.drop(['SSB_CRMSYSTEM_CONTACT_ID', 'target'], axis = 1)
low_variance = variance_threshold_selector(df ,min_variance)
print('columns removed:')
dropped_cols = (df.columns ^ low_variance.columns).values
print(dropped_cols)

#dropped based on minimam variance
dropped = full_data.drop(dropped_cols, axis = 1)

sampled = dropped.sample(frac=1)

X = sampled.drop(['SSB_CRMSYSTEM_CONTACT_ID','target'], axis = 1)
y = sampled['target']
label_encoder = sk.LabelEncoder().fit(y)
y = label_encoder.transform(y)

def train_and_predict(X_train, y_train, X_test, y_test):
    model = LogisticRegression(penalty='l2', solver='liblinear').fit(X_train, y_train)
    preds = (model.predict_proba(X_test)[:,1] >= .5).astype(bool)

    roc_score = roc_auc_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    print(f'ROC: {roc_score}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    return (roc_score, recall, precision)

X_use, X_holdout, y_use, y_holdout = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)


kf = StratifiedKFold(n_splits=5) # Define the split - into 5 folds
roc = 0
prec = 0
rec = 0
for train_index, test_index in kf.split(X_use, y_use):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_use.iloc[train_index], X_use.iloc[test_index]
    y_train, y_test = y_use[train_index], y_use[test_index]

    ro,re,p = train_and_predict(X_train, y_train, X_test, y_test)
    roc += ro
    prec += p
    rec += re

print(f'Avg ROC: {roc/5}')
print(f'Avg Precision: {prec/5}')
print(f'Avg Recall: {rec/5}')


################################################################################
## FiNAL EVAL ##################################################################
################################################################################
