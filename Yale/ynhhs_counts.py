
#%%
# Import libraries
import warnings
import seaborn as sn
import re
import glob
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import os
from tableone import TableOne, load_dataset
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 500)

warnings.filterwarnings('ignore')

#%%

study_pop = pd.read_csv('/path_to_yale/yale_df_processed.csv')
print(len(study_pop))

study_pop['pred_col'] = study_pop['preds_1lead_march31']

study_pop = study_pop[study_pop['pred_col'].notna()]
len(study_pop)

study_pop = study_pop[study_pop['MRN_in_trainDF_single_lead_model']==False]
len(study_pop)

study_pop = study_pop[study_pop['AnyHF_before_inclusion']==0]
len(study_pop)

study_pop = study_pop[study_pop['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG']==False]
len(study_pop)

study_pop = study_pop[study_pop['ECG_during_inpatient_visit']==False]
len(study_pop)

study_pop = study_pop[study_pop['Highest_BNP_before_ECG'].isna() | (study_pop['Highest_BNP_before_ECG'] < 300)]
len(study_pop)

#%%
study_pop['follow_up_time_yrs'] = (pd.to_datetime(study_pop['censor_date']) - pd.to_datetime(study_pop['ECGDate'])).dt.days/365.25
study_pop['age_under_65'] = np.where(study_pop['Age_at_ECG']<65 , 'Age < 65', 'Age >= 65')

#%%
study_pop['screen_positive'] = np.where(study_pop['pred_col']> 0.08, True, False)

study_pop['obesity'] = np.where(study_pop['bmi']>=30, True, False)

#%%
# Make cols for AF and LBBB
ecg = pd.read_csv('/path_to_ecg/ynhhs_ecgs.csv')

study_pop = study_pop.merge(ecg[['fileID', 'Atrial Fibrillation', 'Left Bundle Branch Block']], how = 'left', on = 'fileID')
#%%
study_pop['ESRD'] = np.where(study_pop['gfr']<15, True, False)

#%%
table1_columns = ['Age_at_ECG',
                  'age_under_65',
                'SEX',
                'race_ethnicity_demographics',
                'death',
                'follow_up_time_yrs',
                'screen_positive',
                'AnyHTN_before_inclusion',
                'AnyT2DM_before_inclusion',
                'obesity',
                'Atrial Fibrillation',
                'Left Bundle Branch Block',
                'htn_treatment_before_ecg',
                't2dm_treatment_before_ecg',
                'ESRD',
                'PrimaryHF_after_inclusion',
                'PrimaryHF_or_EchoUnder40',
                'PrimaryHF_or_EchoUnder50',
                'AnyHF_after_inclusion', 
                'AnyHF_or_EchoUnder50',
                'PrimaryAMI_after_inclusion', 
                'PrimarySTROKE_after_inclusion', 
                'PrimaryMACE4',

                   ]

table1_cat = ['SEX',
              'age_under_65',
            'race_ethnicity_demographics',
            'death',
            'screen_positive',
            'AnyHTN_before_inclusion',
            'AnyT2DM_before_inclusion',
            'obesity',
            'Atrial Fibrillation',
            'Left Bundle Branch Block',
            'ESRD',
            'htn_treatment_before_ecg',
            't2dm_treatment_before_ecg',
            'PrimaryHF_after_inclusion',
            'AnyHF_after_inclusion', 
            'PrimaryHF_or_EchoUnder40',
            'PrimaryHF_or_EchoUnder50',
            'PrimaryAMI_after_inclusion', 
            'PrimarySTROKE_after_inclusion', 
            'PrimaryMACE4',
            'AnyHF_or_EchoUnder50',
              ]


nonnormal = ['Age_at_ECG', 
             'follow_up_time_yrs',
             ]

groupby = ['Clean_InstitutionName']

new_order = {"race_ethnicity_demographics": ['White', 'Black', 'Hispanic', 'Asian', 'Others','Missing']}

mytable = TableOne(study_pop, 
                   order = new_order,
                   columns=table1_columns, 
                   categorical=table1_cat, 
                   groupby = groupby, 
                   nonnormal= nonnormal,
                   pval=False)

mytable
#%%