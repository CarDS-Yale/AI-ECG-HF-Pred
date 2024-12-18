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
from sklearn.utils import resample
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 500)

#%%
ukb_pop = pd.read_csv('/path_to_ukb/ukb_df_processed.csv')
print(len(ukb_pop))

ukb_pop['pred_col'] = ukb_pop['preds_1lead_march31']
#%%
ukb_pop['ethnicity_grouped'] = np.where(ukb_pop['ethnicity'] == 1 , 'White', 
                                        np.where(ukb_pop['ethnicity'] == 2 , 'Others', 
                                            np.where(ukb_pop['ethnicity'] == 3 , 'Asian', 
                                                np.where(ukb_pop['ethnicity'] == 4 , 'Black', 
                                                    np.where(ukb_pop['ethnicity'] == 5 , 'Asian', 
                                                        np.where(ukb_pop['ethnicity'] == 6 , 'Others', 'Others' ))))))

# %%
# Clean up the ukb_pop to get to the cohort:

ukb_pop = ukb_pop[ukb_pop['pred_col'].notna()]

# Remove those with earliest any HF date after ECG date (prevalent HF)

datetime_cols = ['birth_date', 'lost_fu_date', 'death_date', 'earliest_date_hf_primary', 'earliest_date_hf_all', 'visit',]

ukb_pop[datetime_cols] = ukb_pop[datetime_cols].apply(pd.to_datetime)

ukb_pop = ukb_pop[ukb_pop['primary_HF'].notna()]
print(len(ukb_pop))

ukb_pop = ukb_pop[ukb_pop['ecg_instance']==2]
print(len(ukb_pop))


ukb_pop = ukb_pop[(ukb_pop['visit'] <= ukb_pop['earliest_date_hf_all']) | (ukb_pop['earliest_date_hf_all'].isna())]
print(len(ukb_pop))

ukb_pop['follow_up_time_yrs'] = (pd.to_datetime("2021-05-20") - ukb_pop['visit']).dt.days/365.25

ukb_pop['screen_positive'] = np.where(ukb_pop['pred_col']> 0.08, True, False)
#%%
table1_columns = ['age',
                  'gender',
                  'ethnicity_grouped',
                  'Death_all_cause',
                  'follow_up_time_yrs',
                  'screen_positive',
                  'primary_HF',
                  'any_HF', 
                  'primary_AMI', 
                  'primary_Stroke', 
                  'primary_MACE'
                   ]

table1_cat = ['gender',
            'ethnicity_grouped',
            'Death_all_cause',
            'screen_positive',
            'primary_HF',
            'any_HF', 
            'primary_AMI', 
            'primary_Stroke', 
            'primary_MACE'
              ]


nonnormal = ['age', 
             'follow_up_time_yrs',
             ]


new_order = {"ethnicity_grouped": ['White', 'Black', 'Asian', 'Others',]}

mytable = TableOne(ukb_pop, 
                   order = new_order,
                   columns=table1_columns, 
                   categorical=table1_cat, 
                   nonnormal= nonnormal,
                   pval=False)

mytable
#%%