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

from Equations.pcp_hf_equation import *

#%%
study_pop = pd.read_csv('/path_to_yale/yale_df_processed.csv')
study_pop_mrns = study_pop['PAT_MRN_ID'].unique().tolist()
print('loaded study_pop')
study_pop_backup = study_pop.copy()

#%%
#QRS Duration
ecg = pd.read_csv('/path_to_ecg/ynhhs_ecgs.csv')

ecg['QRS_Duration'] = pd.to_numeric(ecg['QRS_Duration'], errors='coerce', downcast='integer')
study_pop = study_pop.merge(ecg[['fileID', 'QRS_Duration']], how = 'left', on = 'fileID')

del ecg

study_pop['race_sex_group'] = np.where((study_pop['SEX'] == 'Male') & (study_pop['race_ethnicity_demographics'] == 'White'), 'White Male', 
                                        np.where((study_pop['SEX'] == 'Female') & (study_pop['race_ethnicity_demographics'] == 'White'), 'White Female', 
                                            np.where((study_pop['SEX'] == 'Male') & (study_pop['race_ethnicity_demographics'] == 'Black'), 'Black Male', 
                                                np.where((study_pop['SEX'] == 'Female') & (study_pop['race_ethnicity_demographics'] == 'Black'), 'Black Female', 'Others'))))

#%% 
# Vitals = BMI and BP
all_vitals = pd.read_csv('/path_to_yale/yale_vitals.csv')

all_vitals = all_vitals.merge(study_pop[['PAT_MRN_ID', 'ECGDate']], on = 'PAT_MRN_ID', how = 'left')
all_vitals['RECORDED_TIME'] = pd.to_datetime(all_vitals['RECORDED_TIME'])
all_vitals['ECGDate'] = pd.to_datetime(all_vitals['ECGDate'])

all_vitals['ft_ecg_abs_delta'] = (all_vitals['RECORDED_TIME'] - all_vitals['ECGDate']).dt.days.abs()

all_vitals = all_vitals.sort_values(by = 'ft_ecg_abs_delta', ascending = True)

all_vitals = all_vitals[all_vitals['ft_ecg_abs_delta'] < 730]

wt = all_vitals[all_vitals['DISP_NAME']=='Weight']
ht = all_vitals[all_vitals['DISP_NAME']=='Height']
bp = all_vitals[all_vitals['DISP_NAME']=='BP']


del all_vitals


wt = wt.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')
ht = ht.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')
bp = bp.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')

# merge ht, wt on 'PAT_MRN_ID' but keep all rows from both df, for vars add _ht or _wt as suffix
bmi_joined = pd.merge(ht, wt, on='PAT_MRN_ID', how='outer', suffixes=('_ht', '_wt'))
# convert to float
bmi_joined['MEAS_VALUE_wt'] = bmi_joined['MEAS_VALUE_wt'].astype(float)
bmi_joined['MEAS_VALUE_ht'] = bmi_joined['MEAS_VALUE_ht'].astype(float)
# Calculate BMI and assign it to the 'bmi' column
bmi_joined['bmi'] = ((bmi_joined['MEAS_VALUE_wt'] * 0.0625) / (bmi_joined['MEAS_VALUE_ht'] ** 2)) * 703


bp[['systolic_bp', 'dbp']] = bp['MEAS_VALUE'].str.split('/', expand=True)
bp['systolic_bp'] = bp['systolic_bp'].astype(int)

study_pop = study_pop.merge(bmi_joined[['PAT_MRN_ID', 'bmi']], how = 'left', on = 'PAT_MRN_ID')
study_pop = study_pop.merge(bp[['PAT_MRN_ID', 'systolic_bp']], how = 'left', on = 'PAT_MRN_ID')

study_pop['bmi'] = np.where(study_pop['bmi'] < 10, np.nan, study_pop['bmi'])
study_pop['systolic_bp'] = np.where(study_pop['systolic_bp'] < 60, np.nan, study_pop['systolic_bp'])


del wt, ht, bp
#%% 
# Meds extraction - On treatment for HTN/T2DM Yes/No
meds = pd.read_csv('/path_to_yale/yale_meds.csv')
meds = meds.merge(study_pop[['PAT_MRN_ID', 'ECGDate']], on = 'PAT_MRN_ID', how = 'left')

meds['earliest_any_htn_drug_date'] = pd.to_datetime(meds['earliest_any_htn_drug_date'])
meds['earliest_any_t2dm_drug_date'] = pd.to_datetime(meds['earliest_any_t2dm_drug_date'])
meds['ECGDate'] = pd.to_datetime(meds['ECGDate'])


meds['htn_treatment_before_ecg'] = np.where((meds['earliest_any_htn_drug_date']) < meds['ECGDate'], True, False)
meds['t2dm_treatment_before_ecg'] = np.where((meds['earliest_any_t2dm_drug_date']) < meds['ECGDate'], True, False)

study_pop = study_pop.merge(meds[['PAT_MRN_ID', 'htn_treatment_before_ecg', 't2dm_treatment_before_ecg']], on = 'PAT_MRN_ID', how = 'left')

study_pop['htn_treatment_before_ecg'] = study_pop['htn_treatment_before_ecg'].fillna(False)
study_pop['t2dm_treatment_before_ecg'] = study_pop['t2dm_treatment_before_ecg'].fillna(False)

#%%
# Labs extract
labs1 = pd.read_csv('/path_to_yale/pcp_hf_labs1.csv')
labs2 = pd.read_csv('/path_to_yale/pcp_hf_labs2.csv')
labs3 = pd.read_csv('/path_to_yale/pcp_hf_labs3.csv')

labs = pd.concat([labs1, labs2, labs3])
labs = labs.merge(study_pop[['PAT_MRN_ID', 'ECGDate']], on = 'PAT_MRN_ID', how = 'left')

labs['LAB_DATE'] = pd.to_datetime(labs['LAB_DATE'])
labs['ECGDate'] = pd.to_datetime(labs['ECGDate'])
labs['ft_ecg_abs_delta'] = (labs['LAB_DATE'] - labs['ECGDate']).dt.days.abs()
labs = labs.sort_values(by = 'ft_ecg_abs_delta', ascending = True)
labs = labs[labs['ft_ecg_abs_delta']<730]
    # Keeping values within 2 years 

hdl_vals = ['BKR HDL CHOLESTEROL', 'HDL CHOLESTEROL', 'HIGH DENSITY CHOLESTEROL', 'POC HIGH DENSITY CHOLESTEROL', 'HDL (ABSTRACTED)']
cholesterol_vals = ['BKR CHOLESTEROL', 'CHOLESTEROL', 'CHOLESTEROL, TOTAL']
glucose_vals = ['GLUCOSE', 'BKR GLUCOSE', 'BKR ESTIMATED AVERAGE GLUCOSE']

hdl = labs[labs['COMPONENT_NAME'].isin(hdl_vals)].drop_duplicates(subset = ['PAT_MRN_ID'], keep = 'first').rename(columns={'ORD_NUM_VALUE_CALC': 'hdl_c'})[['PAT_MRN_ID', 'hdl_c']]
cholesterol = labs[labs['COMPONENT_NAME'].isin(cholesterol_vals)].drop_duplicates(subset = ['PAT_MRN_ID'], keep = 'first').rename(columns={'ORD_NUM_VALUE_CALC': 'total_cholesterol'})[['PAT_MRN_ID', 'total_cholesterol']]
glucose = labs[labs['COMPONENT_NAME'].isin(glucose_vals)].drop_duplicates(subset = ['PAT_MRN_ID'], keep = 'first').rename(columns={'ORD_NUM_VALUE_CALC': 'glucose'})[['PAT_MRN_ID', 'glucose']]

study_pop = study_pop.merge(hdl, how = 'left', on = 'PAT_MRN_ID').merge(cholesterol, how = 'left', on = 'PAT_MRN_ID').merge(glucose, how = 'left', on = 'PAT_MRN_ID')

#%%
# Smoking
smoking = pd.read_csv('/path_to_yale/yale_smoking.csv')

study_pop = study_pop.merge(smoking[['PAT_MRN_ID', 'smoking_active']], on = 'PAT_MRN_ID', how = 'left')

study_pop['smoking_active'] = study_pop['smoking_active'].fillna(False)

#%%
study_pop['pcp_hf_risk'] = study_pop.apply(lambda row: calculate_hf_risk(
    age=row['Age_at_ECG'], 
    systolic_bp=row['systolic_bp'], 
    glucose=row['glucose'], 
    total_cholesterol=row['total_cholesterol'], 
    hdl_c=row['hdl_c'], 
    bmi=row['bmi'], 
    qrs_duration=row['QRS_Duration'], 
    smoker=row['smoking_active'], 
    race_sex_group=row['race_sex_group'], 
    treated_bp=row['htn_treatment_before_ecg'], 
    treated_glucose=row['t2dm_treatment_before_ecg']), axis=1)

# %%
study_pop.to_csv('/path_to_yale/yale_df_processed.csv')

