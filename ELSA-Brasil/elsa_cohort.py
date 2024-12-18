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

from Equations.pcp_hf_equation import *
from Equations.prevent_hf_equation import *
#%%
elsa_outcomes = pd.read_stata('/path_to_elsa/outcome.dta')
elsa_baseline_old = pd.read_stata('/path_to_elsa/baseline_elsa.dta')
elsa_lvef = pd.read_excel('/path_to_elsa/elsa_lvef.xls')
elsa_baseline = pd.read_stata('/path_to_elsa/baseline_elsa_updated.dta')
elsa_all_death_2021 = pd.read_stata('/path_to_elsa/elsa_deaths_2021.dta')
elsa_qrs = pd.read_csv('/path_to_elsa/elsa_qrs.csv')
single_lead_preds = pd.read_csv('/path_to_elsa/elsa_1lead_preds.csv')


# %%
hf_cols = ['s_frcic',
       's_dataic', 's_diasic', 's_frcicpreval', 's_frciccat', 's_dataiccat',
       's_diasiccat']


# %%
elsa_outcomes['fileID'] = elsa_outcomes['idelsa'] + '.npy'
elsa_df = single_lead_preds[['fileID', 'preds_1lead_march31']].merge(elsa_outcomes, on = 'fileID', how = 'left')
elsa_df = elsa_df.merge(elsa_baseline, on = 'idelsa', how = 'left')
elsa_df = elsa_df.merge(elsa_lvef[['idelsa', 'ecoa_Teichholz', 'ecoa_Zderived',]], on = 'idelsa', how = 'left')

elsa_df['hf_event_tmp'] = elsa_df['s_frcic']
elsa_df['time_to_hf_event_tmp'] = elsa_df['s_diasiccat']
elsa_df['prevalent_hf_tmp'] = elsa_df['s_frcicpreval']
elsa_df['age'] = elsa_df['idadea']
elsa_df['sex'] = np.where(elsa_df['rcta8']==1 , 'Male', 'Female')
elsa_df['htn'] = elsa_df['a_has2_2']
elsa_df['t2dm'] = elsa_df['a_dm_3']
elsa_df['baseline_lvef'] = elsa_df['ecoa_Teichholz']

elsa_df['race_ethnicity'] = elsa_df['vifa29']


elsa_df['race_ethnicity'] = np.where(elsa_df['race_ethnicity'] == 1 , 'Black', 
                                        np.where(elsa_df['race_ethnicity'] == 2 , 'Pardo', 
                                            np.where(elsa_df['race_ethnicity'] == 3 , 'White', 
                                                np.where(elsa_df['race_ethnicity'] == 4 , 'Asian', 
                                                    np.where(elsa_df['race_ethnicity'] == 5 , 'Others', 
                                                        np.where(elsa_df['race_ethnicity'] == 6 , 'Others', 'Others' ))))))

elsa_df['ami_event'] = elsa_df['s_frciam']
elsa_df['time_to_ami_event'] = elsa_df['s_diasiam']

elsa_df['stroke_event'] = elsa_df['s_frcavc']
elsa_df['time_to_stroke_event'] = elsa_df['s_diasavc']


#%%
elsa_all_death_2021['all_cause_death_till2021'] = elsa_all_death_2021['s_obito_040121']
elsa_all_death_2021['time_to_all_cause_death'] = elsa_all_death_2021['s_tempoobito_040121']

# I want to censor this longer follow-up to the maximum follow-up in the other dataset. 
# So I will replace all values > 1961 days with 1961

elsa_all_death_2021['time_to_all_cause_death_old_censor'] = np.where(elsa_all_death_2021['time_to_all_cause_death']> 1961 , 1961 , elsa_all_death_2021['time_to_all_cause_death'])

# The death flag should be 0 for people if their time_to_all_cause_death_old_censor == 1961 because that means that they might have died after the original censor

elsa_all_death_2021['all_cause_death_old_censor'] = np.where(elsa_all_death_2021['time_to_all_cause_death_old_censor']==1961 , 0, elsa_all_death_2021['all_cause_death_till2021'])

elsa_df = elsa_df.merge(elsa_all_death_2021 , on = 'idelsa', how = 'left')

#%%
# Define MACE and days to MACE

elsa_df["MACE"] = np.where((elsa_df["all_cause_death_old_censor"] == 1) | (elsa_df["stroke_event"] == 1) | (elsa_df["ami_event"] == 1) | (elsa_df["hf_event_tmp"] == 1), 1, 0)

# Create time to MACE - if 1, take the minimum of time_to_censor, time_to_HF, time_to_ami, time_to_stroke, else take time_to_censor

elsa_df["time_to_MACE"] = np.where(elsa_df["MACE"] == 1, elsa_df[["time_to_all_cause_death_old_censor", "time_to_stroke_event", "time_to_ami_event", "time_to_hf_event_tmp"]].min(axis=1), elsa_df["time_to_all_cause_death_old_censor"])


#%%
# Define features for PCP-HF
elsa_df['race_sex_group'] = np.where((elsa_df['sex'] == 'Male') & (elsa_df['race_ethnicity'] == 'White'), 'White Male', 
                                        np.where((elsa_df['sex'] == 'Female') & (elsa_df['race_ethnicity'] == 'White'), 'White Female', 
                                            np.where((elsa_df['sex'] == 'Male') & (elsa_df['race_ethnicity'] == 'Black'), 'Black Male', 
                                                np.where((elsa_df['sex'] == 'Female') & (elsa_df['race_ethnicity'] == 'Black'), 'Black Female', 'Others'))))


# Systolic BP
elsa_df['systolic_bp'] = elsa_df['a_psis']

# BMI
elsa_df['bmi'] = elsa_df['a_imc1']

# Smoking
elsa_df['smoking_active'] = elsa_df['hvsa01']

# Glucose
elsa_df['glucose'] = elsa_df['laba1']

# HDL
elsa_df['hdl_c'] = elsa_df['laba12_2']

# Cholesterol
elsa_df['total_cholesterol'] = elsa_df['laba10_2']

# T2DM treatment
elsa_df['a_antidiabetics'] = elsa_df['a_antidiabetics'].fillna(0)
elsa_df['t2dm_treatment_before_ecg'] = np.where(elsa_df['a_antidiabetics']!=0 , True, False)

# HTN treatment
elsa_df['htn_treatment_before_ecg'] = elsa_df['a_thas2']


#%%
def extract_idelsa(file_path):
    # This splits the path by '/' and takes the last part (filename), 
    # then splits the filename by '.' and takes the first part (idelsa)
    return file_path.split('/')[-1].split('.')[0]

# Apply the function to the entire column of file paths to create a new idelsa column
elsa_qrs['idelsa'] = elsa_qrs['file_path'].apply(extract_idelsa)

elsa_qrs['QRS_Duration'] = elsa_qrs['mean_QRS_duration']


elsa_df = elsa_df.merge(elsa_qrs[['idelsa', 'QRS_Duration']], how = 'left', on = 'idelsa')
#%%
elsa_df['pcp_hf_risk'] = elsa_df.apply(lambda row: calculate_hf_risk(
    age=row['age'], 
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

pcp_hf_cols = ['pcp_hf_risk','age', 'systolic_bp', 'glucose','total_cholesterol','hdl_c','bmi','QRS_Duration', 'smoking_active', 'race_sex_group', 'htn_treatment_before_ecg', 't2dm_treatment_before_ecg']


# %%
# Remove patients with prevalent HF
elsa_df = elsa_df[elsa_df['s_frcicpreval']==0]
elsa_df = elsa_df[(elsa_df['baseline_lvef'] > 50) | elsa_df['baseline_lvef'].isna()]

# EDA for what the date to censor looks like: 
# Filter for people with HF and check the dates
elsa_df[elsa_df['s_frcic']==1]['s_diasiccat'].value_counts(bins = [0,200,500,1000,2000]).sort_index()


# Define obesity using BMI
elsa_df['obesity'] = np.where(elsa_df['bmi']>=30, True, False)

elsa_df['ihd'] = elsa_df['s_frciampreval']
# %%

table1_columns = ['age',
                  'sex',
                  'race_ethnicity',
                  'all_cause_death_old_censor',
                  'screen_positive',
                  'hf_event_tmp',
                  'ami_event', 
                  'stroke_event', 
                  'MACE',
                  'htn',
                    't2dm', 
                    'obesity',
                    'ihd',
                    'htn_treatment_before_ecg',
                  't2dm_treatment_before_ecg',
                  
                   ]

table1_cat = ['sex',
            'race_ethnicity',
            'all_cause_death_old_censor',
            'screen_positive',
            'hf_event_tmp',
              'ami_event', 
              'stroke_event',
             'MACE',
                'htn',
                't2dm',
                'obesity',
                'ihd',
                'htn_treatment_before_ecg',
                't2dm_treatment_before_ecg',
                
              ]


nonnormal = ['age', 
       #       'follow_up_time_yrs',
             ]


# new_order = {"ethnicity_grouped": ['White', 'Black', 'Asian', 'Others',]}

mytable = TableOne(elsa_df, 
              #      order = new_order,
                   columns=table1_columns, 
                   categorical=table1_cat, 
                   nonnormal= nonnormal,
                   pval=False)

mytable


#%%
# Oct 28 adding PREVENT scores
elsa_labs = pd.read_stata('/path_to_elsa/eGFR.dta')
# Rename a_ckdepi_r_2 to GFR
elsa_labs = elsa_labs.rename(columns={'a_ckdepi_r_2': 'eGFR'})

elsa_df = elsa_df.merge(elsa_labs[['idelsa', 'eGFR']], on = 'idelsa', how = 'left')

# Make new col sex_group with Men/Women values based on sex col in elsa_df with Male/Female values

elsa_df['sex_group'] = elsa_df['sex'].replace({'Male': 'Men', 'Female': 'Women'})

# Calculate the PREVENT score
elsa_df['prevent_hf_10yr_risk'] = elsa_df.apply(lambda row: calculate_prevent_hf_10yr_risk(
    age=row['age'], 
    sbp=row['systolic_bp'], 
    diabetes=row['t2dm'],
    current_smoker=row['smoking_active'], 
    bmi=row['bmi'], 
    gfr=row['eGFR'],
    sex_group=row['sex_group'], #TODO
    antihypertensive_med=row['htn_treatment_before_ecg'], 
    ), axis=1)

elsa_df['prevent_hf_risk_adjusted'] = elsa_df['prevent_hf_10yr_risk'] * elsa_df['time_to_all_cause_death_old_censor'] / 3650 #TODO

elsa_df.to_csv('/path_to_elsa/elsa_df_processed.csv')

# %%