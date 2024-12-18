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
import sys
from sys import getsizeof

pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 500)

warnings.filterwarnings('ignore')

from Equations.prevent_hf_equation import *


#%%
study_pop = pd.read_csv('/path_to_yale/yale_df_processed.csv')
study_pop_mrns = study_pop['PAT_MRN_ID'].unique().tolist()
print('loaded study_pop')
study_pop_backup = study_pop.copy()


'''
Parameters:
[x] diabetes (bool): If the individual has diabetes (True/False).
[-] gfr (float): Glomerular filtration rate.
[x] age (float): The age of the individual.
[x] sbp (float): Systolic blood pressure.
[x] current_smoker (bool): Smoking status (True if current smoker, False otherwise).
[x] bmi (float): Body mass index.
[x] sex_group (str): 'Men' or 'Women' to determine which coefficients to use.
[x] antihypertensive_med (bool): If the individual uses antihypertensive medication (True/False).

'''

#%%
# Labs extract
labs1 = pd.read_csv('/path_to_yale/prevent_hf_labs1.csv')
labs2 = pd.read_csv('/path_to_yale/prevent_hf_labs2.csv')
labs3 = pd.read_csv('/path_to_yale/prevent_hf_labs3.csv')

labs = pd.concat([labs1, labs2, labs3])
labs = labs.merge(study_pop[['PAT_MRN_ID', 'ECGDate']], on = 'PAT_MRN_ID', how = 'left')

labs['LAB_DATE'] = pd.to_datetime(labs['LAB_DATE'])
labs['ECGDate'] = pd.to_datetime(labs['ECGDate'])
labs['ft_ecg_abs_delta'] = (labs['LAB_DATE'] - labs['ECGDate']).dt.days.abs()
labs = labs.sort_values(by = 'ft_ecg_abs_delta', ascending = True)

#%%

labs = labs[labs['ft_ecg_abs_delta']<730]
    # Keeping values within 2 years 

gfr = labs.copy()

#%%
gfr = gfr.drop_duplicates(subset = ['PAT_MRN_ID'], keep = 'first').rename(columns={'ORD_NUM_VALUE_CALC': 'gfr'})[['PAT_MRN_ID', 'gfr']]
study_pop = study_pop.merge(gfr, how = 'left', on = 'PAT_MRN_ID')

#%%
# Make sex_group Men/Women based on SEX being Male or Otherwise
study_pop['sex_group'] = study_pop['SEX'].apply(lambda x: 'Men' if x == 'Male' else 'Women')


#%%
# Calculate the 10-year PREVENT heart failure risk for each individual
study_pop['prevent_hf_10yr_risk'] = study_pop.apply(lambda row: calculate_prevent_hf_10yr_risk(
    age=row['Age_at_ECG'], 
    sbp=row['systolic_bp'], 
    diabetes=row['AnyT2DM_before_inclusion'],
    current_smoker=row['smoking_active'], 
    bmi=row['bmi'], 
    gfr=row['gfr'],
    sex_group=row['sex_group'],
    antihypertensive_med=row['htn_treatment_before_ecg'], 
    ), axis=1)


#%%
study_pop['prevent_hf_risk_adjusted'] = study_pop['prevent_hf_10yr_risk'] * study_pop['time_to_censor'] / 3650


# %%
study_pop.to_csv('/path_to_yale/yale_df_processed.csv')