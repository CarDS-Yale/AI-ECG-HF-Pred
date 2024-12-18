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


#%%
# Basic function

def read_full(path, filename):
    df = pd.read_csv(path + filename,
                             sep='\t', on_bad_lines='skip', low_memory=False,)
    return df


#%%
# Paths:
path = '/path_to_yale_ehr/'

hosp_enc_df_cols = ['PAT_MRN_ID', 'PAT_ENC_CSN_ID', 'BIRTH_DATE', 'DEATH_DATE', 'HOSP_ADMSN_DATE', 'ACCT_BASECLS_HA']
out_enc_df_cols = ['PAT_MRN_ID', 'PAT_ENC_CSN_ID', 'CONTACT_DATE']
dx_df_cols = ['PAT_MRN_ID', 'PAT_ENC_CSN_ID', 'CALC_DX_DATE', 'DX_SOURCE', 'PRIMARY_YN', 'CURRENT_ICD10_LIST', 'DX_NAME']

#%%
# Dictionaries of ICD terms

hf_icd_list = ['I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509', 'Z9581', 'I0981',
            #   '428','4280','4281','4289'
                      ]

ami_icd_list = ['I21', # Acute myocardial infarction
            'I22', # Subsequent myocardial infarction
            'I23', # Certain current complications following acute myocardial infarction
            'I240', # Acute coronary thrombosis not resulting in myocardial infarction
            'I248', # Other forms of acute ischemic heart disease
            'I249' # Acute ischemic heart disease, unspecified
            # '410', '4110', '4111', '4118'
            ]

ihd_icd_list = ['I20', 'I200', 'I208', 'I209', 'I21', 'I210', 'I211', 'I212', 'I213',
            'I214', 'I219', 'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I23', 'I230',
            'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 'I24', 'I240', 'I241',
            'I248', 'I249', 'I25', 'I250', 'I251', 'I252', 'I255', 'I256', 'I258', 'I259',
            'Z951', 'Z955',
            # '410', '4109', '411', '4119', '412', '4129', '413', '4139', '414', '4140',
            # '4148', '4149'
            ]

stroke_icd_list=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459', 'I63', 
             'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638', 'I639', 'I64',
             'I65', 'I650', 'I651', 'I652', 'I653', 'I658', 'I659', 'I66', 'I660',
             'I661', 'I662', 'I663', 'I664', 'I668', 'I669', 'I672', 'I693', 'I694',
             
            #  '433', '4330', '4331', '4332', '4333', '4338', '4339', '434', '4340',
            #  '4341', '4349', '435', '4359', '437', '4370', '4371'
             ]


htn_icd_list = ['I10', 'I11', 'I110', 'I119', 'I12', 'I120', 'I129', 'I13', 'I130', 'I131',
            'I132', 'I139', 'I674', 'O10', 'O100', 'O101', 'O102', 'O103', 'O109', 'O11',      
            # '401', '4010', '4011', '4019', '402', '4020', '4021', '4029', '403', '4030',
            # '4031', '4039', '404', '4040', '4041', '4049', '6420', '6422', '6427', '6429'
            ]

dm_icd_list = ['E10', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108',
           'E109', 'E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117',
           'E118', 'E119', 'E12', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126',
           'E127', 'E128', 'E129', 'E13', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135',
           'E136', 'E137', 'E138', 'E139', 'E14', 'E140', 'E141', 'E142', 'E143', 'E144',
           'E145', 'E146', 'E147', 'E148', 'E149', 'O240', 'O241', 'O242', 'O243', 'O249',
        #    '250', '2500', '25000', '25001', '25009', '2501', '25010', '25011', '25019', '2502',
        #    '25020', '25021', '25029', '2503', '2504', '2505', '2506', '2507', '2509', '25090',
        #    '25091', '25099', '6480'
           ]

t2dm_icd_list = ['E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118',
             'E119', 'O241',
             '25000', '25010', '25020', '25090']

def add_dot(code):
    if len(code) > 3:
        return code[:3] + '.' + code[3:]
    else:
        return code

hf_icd_list = [add_dot(code) for code in hf_icd_list]
ami_icd_list = [add_dot(code) for code in ami_icd_list]
ihd_icd_list = [add_dot(code) for code in ihd_icd_list]
stroke_icd_list = [add_dot(code) for code in stroke_icd_list]
htn_icd_list = [add_dot(code) for code in htn_icd_list]
dm_icd_list = [add_dot(code) for code in dm_icd_list]
t2dm_icd_list = [add_dot(code) for code in t2dm_icd_list]

outcome_icd_codes = hf_icd_list + ami_icd_list + ihd_icd_list + stroke_icd_list
dm_htn_icd_codes = htn_icd_list + dm_icd_list


#%%
# To pick the earliest encounter of a patients in the healthcare system:

enc_dates = read_full(path, 'Hospital_Enc.txt')[['PAT_MRN_ID', 'HOSP_ADMSN_DATE']].rename(columns={'HOSP_ADMSN_DATE': 'enc_date'})

enc_dates['enc_date'] = pd.to_datetime(enc_dates['enc_date'])

earliest_enc = enc_dates.sort_values(by = 'enc_date', ascending = True).drop_duplicates(subset=['PAT_MRN_ID'], keep = 'first')
earliest_enc = earliest_enc.rename(columns={'enc_date': 'earliest_enc_date', 'PAT_MRN_ID':'MRN'})

latest_enc = enc_dates.sort_values(by = 'enc_date', ascending = False).drop_duplicates(subset=['PAT_MRN_ID'], keep = 'first')
latest_enc = latest_enc.rename(columns={'enc_date': 'latest_enc_date', 'PAT_MRN_ID':'MRN'})

# To add 1 year to this date:
earliest_enc['after_blanking_year'] = earliest_enc['earliest_enc_date'] + pd.DateOffset(years=1)


#%%
ecg = pd.read_csv('/path_to_ecg/ynhhs_ecgs.csv', low_memory=False)
ecg['ECGDate'] = pd.to_datetime(ecg['ECGDate'])
ecg_ppl_mrn_list = ecg['MRN'].unique().tolist()

#%%
ecg = ecg.merge(earliest_enc, how = 'left', on = 'MRN')
ecg = ecg.merge(latest_enc, how = 'left', on = 'MRN')


# Keep the first ECG after after_blanking_year.
# Filter for ECGs after the after_blanking_year. Then sort by ECGDate and drop duplicates. 

ecg_after_blanking = ecg[ecg['ECGDate'] > ecg['after_blanking_year']].sort_values(by = 'ECGDate', ascending = True).drop_duplicates(subset=['MRN'], keep = 'first')
ecg_after_blanking['PAT_MRN_ID'] = ecg_after_blanking['MRN'].copy()


study_pop = ecg_after_blanking[['MRN', 'PAT_MRN_ID', 'ECGDate', 'after_blanking_year', 'fileID',  'fails_to_load_25jan2024', 'YearMonth', 'Year', 'Clean_InstitutionName', 'external_site_ecg', 'earliest_enc_date', 'latest_enc_date' ]]

#%%

# Clean up demographics and some person-level flags

pt_df = read_full(path, 'Demographics.txt')[['PAT_MRN_ID', 'BIRTH_DATE', 'DEATH_DATE', 'SEX', 'PATIENT_ETHNICITY', 'PATIENT_RACE_ALL',]]

study_pop = study_pop.merge(pt_df, how ='left', on = 'PAT_MRN_ID')

# Categorize Race and Ethnicity
def race_categorize(row):
    if row['PATIENT_RACE_ALL'] == 'White or Caucasian':
        return 'White'
    if row['PATIENT_RACE_ALL'] == 'White':
        return 'White'
    elif row['PATIENT_RACE_ALL'] == 'Black or African American':
        return 'Black'
    elif row['PATIENT_RACE_ALL'] == 'Unknown':
        return 'Missing'
    elif row['PATIENT_RACE_ALL'] == 'Not Listed':
        return 'Missing'
    elif row['PATIENT_RACE_ALL'] == 'Asian':
        return 'Asian'
    return 'Others'

study_pop['race_categorize_demographics'] = study_pop.apply(
    lambda row: race_categorize(row), axis=1)

# Combine race and ethnicity in one column
def ethnicity_categorize(row):
    if row['PATIENT_ETHNICITY'] == 'Hispanic or Latino':
        return 'Hispanic'
    if row['PATIENT_ETHNICITY'] == 'Hispanic or Latina/o/x':
        return 'Hispanic'
    return row['race_categorize_demographics']


study_pop['race_ethnicity_demographics'] = study_pop.apply(
    lambda row: ethnicity_categorize(row), axis=1)

study_pop['ethnicity_categorize_demographics'] = np.where(study_pop['race_ethnicity_demographics']=='Hispanic', 'Hispanic', 'Non Hispanic')

study_pop = study_pop.drop(columns=['PATIENT_RACE_ALL', 'PATIENT_ETHNICITY'], axis=1)

study_pop['SEX'] = np.where(pd.isna(study_pop['SEX']), 'Unknown', study_pop['SEX'])

# Age in years at ECGDate
study_pop['BIRTH_DATE'] = pd.to_datetime(study_pop['BIRTH_DATE'])
study_pop['ECGDate'] = pd.to_datetime(study_pop['ECGDate'])

study_pop['Age_at_ECG'] = (study_pop['ECGDate'] - study_pop['BIRTH_DATE']).dt.days / 365.25
study_pop['Age_at_ECG'] = study_pop['Age_at_ECG'].round(2)

#%%
combo_dx = pd.read_csv(path + 'diagnosis_codes.csv', low_memory = False)
combo_dx['CALC_DX_DATE'] = pd.to_datetime(combo_dx['CALC_DX_DATE'])

combo_dx = combo_dx.merge(ecg_after_blanking[['PAT_MRN_ID', 'ECGDate']])
print(len(combo_dx))

combo_dx_before = combo_dx[pd.to_datetime(combo_dx['CALC_DX_DATE']) < combo_dx['ECGDate']]
print(len(combo_dx_before))

combo_dx_after = combo_dx[pd.to_datetime(combo_dx['CALC_DX_DATE']) > combo_dx['ECGDate']]
print(len(combo_dx_after))



#%%
# Identify disease rows in problem list with disease BEFORE inclusion date

# Make lists of PATIENTS with disease BEFORE inclusion date
# Define diagnoses

# AnyHF
combo_dx_before["AnyHF"] = np.where(combo_dx_before["CURRENT_ICD10_LIST"].isin(hf_icd_list), 1, 0)
any_hf_before_inclusion_ids = combo_dx_before[combo_dx_before["AnyHF"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_before["AnyHF"].value_counts())
print(len(any_hf_before_inclusion_ids))

# AnyAMI
combo_dx_before["AnyAMI"] = np.where(combo_dx_before["CURRENT_ICD10_LIST"].isin(ami_icd_list), 1, 0)
any_ami_before_inclusion_ids = combo_dx_before[combo_dx_before["AnyAMI"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_before["AnyAMI"].value_counts())
print(len(any_ami_before_inclusion_ids))

# AnyIHD
combo_dx_before["AnyIHD"] = np.where(combo_dx_before["CURRENT_ICD10_LIST"].isin(ihd_icd_list), 1, 0)
any_ihd_before_inclusion_ids = combo_dx_before[combo_dx_before["AnyIHD"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_before["AnyIHD"].value_counts())
print(len(any_ihd_before_inclusion_ids))

# AnySTROKE
combo_dx_before["AnySTROKE"] = np.where(combo_dx_before["CURRENT_ICD10_LIST"].isin(stroke_icd_list), 1, 0)
any_stroke_before_inclusion_ids = combo_dx_before[combo_dx_before["AnySTROKE"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_before["AnySTROKE"].value_counts())
print(len(any_stroke_before_inclusion_ids))

#####
combo_dx_before_sorted = combo_dx_before.sort_values(by=['PAT_MRN_ID', 'CALC_DX_DATE'])
# Then, define your aggregation dictionary
agg_dict = {
    "AnyHF": 'max',
    "AnyAMI": 'max',
    "AnyIHD": 'max',
    "AnySTROKE": 'max'
}

# Group by 'PAT_MRN_ID' and aggregate
combo_dx_before_aggregated = combo_dx_before_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Now, you have maximum (or most recent) values for each diagnosis flag and the first DX date
# If you want to make columns for each specific diagnosis' first date, you would do so in separate groupby and join steps
for diag in ["AnyHF", "AnyAMI", "AnyIHD", "AnySTROKE"]:
    first_date_df = combo_dx_before_sorted[combo_dx_before_sorted[diag] == 1].groupby('PAT_MRN_ID')['CALC_DX_DATE'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'CALC_DX_DATE': f'{diag}_before_inclusion_dx_date'})
    combo_dx_before_aggregated = pd.merge(combo_dx_before_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

for diag in ["AnyHF", "AnyAMI", "AnyIHD", "AnySTROKE"]:
    combo_dx_before_aggregated = combo_dx_before_aggregated.rename(columns={f'{diag}': f'{diag}_before_inclusion'})


# Identify disease rows in problem list with disease AFTER inclusion date - Identify the FIRST mention of disease AFTER inclusion date

# Define diagnoses

# AnyHF

combo_dx_after["AnyHF_after_inclusion"] = np.where(combo_dx_after["CURRENT_ICD10_LIST"].isin(hf_icd_list), 1, 0)
any_hf_after_inclusion_ids = combo_dx_after[combo_dx_after["AnyHF_after_inclusion"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_after["AnyHF_after_inclusion"].value_counts())
print(len(any_hf_after_inclusion_ids))

# AnyAMI
combo_dx_after["AnyAMI_after_inclusion"] = np.where(combo_dx_after["CURRENT_ICD10_LIST"].isin(ami_icd_list), 1, 0)
any_ami_after_inclusion_ids = combo_dx_after[combo_dx_after["AnyAMI_after_inclusion"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_after["AnyAMI_after_inclusion"].value_counts())
print(len(any_ami_after_inclusion_ids))

# AnySTROKE
combo_dx_after["AnySTROKE_after_inclusion"] = np.where(combo_dx_after["CURRENT_ICD10_LIST"].isin(stroke_icd_list), 1, 0)
any_stroke_after_inclusion_ids = combo_dx_after[combo_dx_after["AnySTROKE_after_inclusion"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_after["AnySTROKE_after_inclusion"].value_counts())
print(len(any_stroke_after_inclusion_ids))


combo_dx_after_sorted = combo_dx_after.sort_values(by=['PAT_MRN_ID', 'CALC_DX_DATE'])

# Then, define your aggregation dictionary
agg_dict = {
    "AnyHF_after_inclusion": 'max',
    "AnyAMI_after_inclusion": 'max',
    "AnySTROKE_after_inclusion": 'max'
}

# Group by 'PAT_MRN_ID' and aggregate
combo_dx_after_aggregated = combo_dx_after_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Now, you have maximum (or most recent) values for each diagnosis flag and the first DX date
# If you want to make columns for each specific diagnosis' first date, you would do so in separate groupby and join steps
for diag in ["AnyHF_after_inclusion", "AnyAMI_after_inclusion", "AnySTROKE_after_inclusion"]:
    first_date_df = combo_dx_after_sorted[combo_dx_after_sorted[diag] == 1].groupby('PAT_MRN_ID')['CALC_DX_DATE'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'CALC_DX_DATE': f'{diag}_dx_date'})
    combo_dx_after_aggregated = pd.merge(combo_dx_after_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

#%%
primary_combo_dx = combo_dx[(combo_dx['PRIMARY_YN']=='Y')& (combo_dx['DX_SOURCE']=='HOSPITAL_BILLING_DX')] # Keep the primary diagnosis only
print(len(primary_combo_dx))

primary_combo_dx_before = primary_combo_dx[pd.to_datetime(primary_combo_dx['CALC_DX_DATE']) < primary_combo_dx['ECGDate']]
print(len(primary_combo_dx_before))

primary_combo_dx_after = primary_combo_dx[pd.to_datetime(primary_combo_dx['CALC_DX_DATE']) > primary_combo_dx['ECGDate']]
print(len(primary_combo_dx_after))
#%%

# PrimaryHF
primary_combo_dx_before["PrimaryHF"] = np.where(primary_combo_dx_before["CURRENT_ICD10_LIST"].isin(hf_icd_list), 1, 0)
primary_hf_before_inclusion_ids = primary_combo_dx_before[primary_combo_dx_before["PrimaryHF"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_before["PrimaryHF"].value_counts())
print(len(primary_hf_before_inclusion_ids))

# PrimaryAMI
primary_combo_dx_before["PrimaryAMI"] = np.where(primary_combo_dx_before["CURRENT_ICD10_LIST"].isin(ami_icd_list), 1, 0)
primary_ami_before_inclusion_ids = primary_combo_dx_before[primary_combo_dx_before["PrimaryAMI"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_before["PrimaryAMI"].value_counts())
print(len(primary_ami_before_inclusion_ids))

# PrimaryIHD
primary_combo_dx_before["PrimaryIHD"] = np.where(primary_combo_dx_before["CURRENT_ICD10_LIST"].isin(ihd_icd_list), 1, 0)
primary_ihd_before_inclusion_ids = primary_combo_dx_before[primary_combo_dx_before["PrimaryIHD"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_before["PrimaryIHD"].value_counts())
print(len(primary_ihd_before_inclusion_ids))

# PrimarySTROKE
primary_combo_dx_before["PrimarySTROKE"] = np.where(primary_combo_dx_before["CURRENT_ICD10_LIST"].isin(stroke_icd_list), 1, 0)
primary_stroke_before_inclusion_ids = primary_combo_dx_before[primary_combo_dx_before["PrimarySTROKE"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_before["PrimarySTROKE"].value_counts())
print(len(primary_stroke_before_inclusion_ids))

#####
primary_combo_dx_before_sorted = primary_combo_dx_before.sort_values(by=['PAT_MRN_ID', 'CALC_DX_DATE'])
# Then, define your aggregation dictionary
agg_dict = {
    "PrimaryHF": 'max',
    "PrimaryAMI": 'max',
    "PrimaryIHD": 'max',
    "PrimarySTROKE": 'max'
}

# Group by 'PAT_MRN_ID' and aggregate
primary_combo_dx_before_aggregated = primary_combo_dx_before_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Now, you have maximum (or most recent) values for each diagnosis flag and the first DX date
# If you want to make columns for each specific diagnosis' first date, you would do so in separate groupby and join steps
for diag in ["PrimaryHF", "PrimaryAMI", "PrimaryIHD", "PrimarySTROKE"]:
    first_date_df = primary_combo_dx_before_sorted[primary_combo_dx_before_sorted[diag] == 1].groupby('PAT_MRN_ID')['CALC_DX_DATE'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'CALC_DX_DATE': f'{diag}_before_inclusion_dx_date'})
    primary_combo_dx_before_aggregated = pd.merge(primary_combo_dx_before_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

for diag in ["PrimaryHF", "PrimaryAMI", "PrimaryIHD", "PrimarySTROKE"]:
    primary_combo_dx_before_aggregated = primary_combo_dx_before_aggregated.rename(columns={f'{diag}': f'{diag}_before_inclusion'})

# Identify disease rows in problem list with disease AFTER inclusion date - Identify the FIRST mention of disease AFTER inclusion date

# Define diagnoses

# PrimaryHF

primary_combo_dx_after["PrimaryHF_after_inclusion"] = np.where(primary_combo_dx_after["CURRENT_ICD10_LIST"].isin(hf_icd_list), 1, 0)
primary_hf_after_inclusion_ids = primary_combo_dx_after[primary_combo_dx_after["PrimaryHF_after_inclusion"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_after["PrimaryHF_after_inclusion"].value_counts())
print(len(primary_hf_after_inclusion_ids))

# PrimaryAMI
primary_combo_dx_after["PrimaryAMI_after_inclusion"] = np.where(primary_combo_dx_after["CURRENT_ICD10_LIST"].isin(ami_icd_list), 1, 0)
primary_ami_after_inclusion_ids = primary_combo_dx_after[primary_combo_dx_after["PrimaryAMI_after_inclusion"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_after["PrimaryAMI_after_inclusion"].value_counts())
print(len(primary_ami_after_inclusion_ids))

# PrimarySTROKE
primary_combo_dx_after["PrimarySTROKE_after_inclusion"] = np.where(primary_combo_dx_after["CURRENT_ICD10_LIST"].isin(stroke_icd_list), 1, 0)
primary_stroke_after_inclusion_ids = primary_combo_dx_after[primary_combo_dx_after["PrimarySTROKE_after_inclusion"] == 1]["PAT_MRN_ID"].unique()
print(primary_combo_dx_after["PrimarySTROKE_after_inclusion"].value_counts())
print(len(primary_stroke_after_inclusion_ids))


primary_combo_dx_after_sorted = primary_combo_dx_after.sort_values(by=['PAT_MRN_ID', 'CALC_DX_DATE'])

# Then, define your aggregation dictionary
agg_dict = {
    "PrimaryHF_after_inclusion": 'max',
    "PrimaryAMI_after_inclusion": 'max',
    "PrimarySTROKE_after_inclusion": 'max'
}

# Group by 'PAT_MRN_ID' and aggregate
primary_combo_dx_after_aggregated = primary_combo_dx_after_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Now, you have maximum (or most recent) values for each diagnosis flag and the first DX date
# If you want to make columns for each specific diagnosis' first date, you would do so in separate groupby and join steps
for diag in ["PrimaryHF_after_inclusion", "PrimaryAMI_after_inclusion", "PrimarySTROKE_after_inclusion"]:
    first_date_df = primary_combo_dx_after_sorted[primary_combo_dx_after_sorted[diag] == 1].groupby('PAT_MRN_ID')['CALC_DX_DATE'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'CALC_DX_DATE': f'{diag}_dx_date'})
    primary_combo_dx_after_aggregated = pd.merge(primary_combo_dx_after_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

#%%
# Merge the AnyDiagnosis and PrimaryDiagnosis (Before and After, along with their dates) to study_pop
study_pop_backup1 = study_pop.copy()

study_pop = study_pop.merge(combo_dx_before_aggregated, on = 'PAT_MRN_ID', how = 'left')
study_pop = study_pop.merge(combo_dx_after_aggregated, on = 'PAT_MRN_ID', how = 'left')


study_pop = study_pop.merge(primary_combo_dx_before_aggregated, on = 'PAT_MRN_ID', how = 'left')
study_pop = study_pop.merge(primary_combo_dx_after_aggregated, on = 'PAT_MRN_ID', how = 'left')


#%%
# For HTN and T2DM

combo_dx_htn_dm = pd.read_csv(path + 'combo_dx_htn_dm.csv', usecols = ['PAT_MRN_ID', 'CALC_DX_DATE', 'CURRENT_ICD10_LIST'])
combo_dx_htn_dm['CALC_DX_DATE'] = pd.to_datetime(combo_dx_htn_dm['CALC_DX_DATE'])

combo_dx_htn_dm = combo_dx_htn_dm.merge(ecg_after_blanking[['PAT_MRN_ID', 'ECGDate']])
print(len(combo_dx_htn_dm))

combo_dx_htn_dm_before = combo_dx_htn_dm[pd.to_datetime(combo_dx_htn_dm['CALC_DX_DATE']) < combo_dx_htn_dm['ECGDate']]
print(len(combo_dx_htn_dm_before))

# Make lists of PATIENTS with disease BEFORE inclusion date
# Define diagnoses

# AnyHTN
combo_dx_htn_dm_before["AnyHTN"] = np.where(combo_dx_htn_dm_before["CURRENT_ICD10_LIST"].isin(htn_icd_list), 1, 0)
any_hf_before_inclusion_ids = combo_dx_htn_dm_before[combo_dx_htn_dm_before["AnyHTN"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_htn_dm_before["AnyHTN"].value_counts())
print(len(any_hf_before_inclusion_ids))

# AnyT2DM
combo_dx_htn_dm_before["AnyT2DM"] = np.where(combo_dx_htn_dm_before["CURRENT_ICD10_LIST"].str.contains('|'.join(t2dm_icd_list), case = False, na = False), 1, 0)
any_ami_before_inclusion_ids = combo_dx_htn_dm_before[combo_dx_htn_dm_before["AnyT2DM"] == 1]["PAT_MRN_ID"].unique()
print(combo_dx_htn_dm_before["AnyT2DM"].value_counts())
print(len(any_ami_before_inclusion_ids))

combo_dx_htn_dm_before_sorted = combo_dx_htn_dm_before.sort_values(by=['PAT_MRN_ID', 'CALC_DX_DATE'])
# Then, define your aggregation dictionary
agg_dict = {
    "AnyHTN": 'max',
    "AnyT2DM": 'max'
}

# Group by 'PAT_MRN_ID' and aggregate
combo_dx_htn_dm_before_aggregated = combo_dx_htn_dm_before_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Now, you have maximum (or most recent) values for each diagnosis flag and the first DX date
# If you want to make columns for each specific diagnosis' first date, you would do so in separate groupby and join steps
for diag in ["AnyHTN", "AnyT2DM"]:
    first_date_df = combo_dx_htn_dm_before_sorted[combo_dx_htn_dm_before_sorted[diag] == 1].groupby('PAT_MRN_ID')['CALC_DX_DATE'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'CALC_DX_DATE': f'{diag}_before_inclusion_dx_date'})
    combo_dx_htn_dm_before_aggregated = pd.merge(combo_dx_htn_dm_before_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

for diag in ["AnyHTN", "AnyT2DM"]:
    combo_dx_htn_dm_before_aggregated = combo_dx_htn_dm_before_aggregated.rename(columns={f'{diag}': f'{diag}_before_inclusion'})


study_pop = study_pop.merge(combo_dx_htn_dm_before_aggregated, on = 'PAT_MRN_ID', how = 'left')


#%%
diagnosis_cols_to_fill_with_0s = ['AnyHF_before_inclusion','AnyAMI_before_inclusion', 'AnyIHD_before_inclusion', 'AnySTROKE_before_inclusion',
                                    'AnyHF_after_inclusion', 'AnyAMI_after_inclusion', 'AnySTROKE_after_inclusion', 
                                    'PrimaryHF_before_inclusion', 'PrimaryAMI_before_inclusion', 'PrimaryIHD_before_inclusion', 'PrimarySTROKE_before_inclusion',
                                    'PrimaryHF_after_inclusion', 'PrimaryAMI_after_inclusion', 'PrimarySTROKE_after_inclusion',
                                    'AnyHTN_before_inclusion' , 'AnyT2DM_before_inclusion']  

study_pop[diagnosis_cols_to_fill_with_0s] = study_pop[diagnosis_cols_to_fill_with_0s].fillna(0)


#%%

# Define date of censor and death
datetime_cols = ['ECGDate', 'after_blanking_year', 'earliest_enc_date', 'latest_enc_date', 'BIRTH_DATE', 'DEATH_DATE', 
                'AnyHF_before_inclusion_dx_date', 'AnyAMI_before_inclusion_dx_date', 'AnyIHD_before_inclusion_dx_date', 'AnySTROKE_before_inclusion_dx_date',
                'AnyHF_after_inclusion_dx_date', 'AnyAMI_after_inclusion_dx_date', 'AnySTROKE_after_inclusion_dx_date',
                'PrimaryHF_before_inclusion_dx_date', 'PrimaryAMI_before_inclusion_dx_date', 'PrimaryIHD_before_inclusion_dx_date', 'PrimarySTROKE_before_inclusion_dx_date', 
                'PrimaryHF_after_inclusion_dx_date', 'PrimaryAMI_after_inclusion_dx_date', 'PrimarySTROKE_after_inclusion_dx_date']

study_pop[datetime_cols] = study_pop[datetime_cols].apply(pd.to_datetime)



# Putting Sept 15, 2023 as the censor date. Previously, by mistake, I had added the latest_enc_date as the censor date for these people
study_pop['censor_date'] = study_pop['DEATH_DATE'].fillna(pd.to_datetime('2023-09-15'))

study_pop["time_to_censor"] = (study_pop["censor_date"] - study_pop["ECGDate"]).dt.days

study_pop["death"] = np.where(study_pop["DEATH_DATE"].isnull(), 0, 1)

# Define time to these events
study_pop["AnyHF_after_inclusion_dx_date"] = study_pop["AnyHF_after_inclusion_dx_date"].fillna(study_pop["censor_date"])
study_pop["AnyAMI_after_inclusion_dx_date"] = pd.to_datetime(study_pop["AnyAMI_after_inclusion_dx_date"]).fillna(study_pop["censor_date"])
study_pop["AnySTROKE_after_inclusion_dx_date"] = pd.to_datetime(study_pop["AnySTROKE_after_inclusion_dx_date"]).fillna(study_pop["censor_date"])

study_pop["time_to_AnyHF"] = (pd.to_datetime(study_pop["AnyHF_after_inclusion_dx_date"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_AnyAMI"] = (pd.to_datetime(study_pop["AnyAMI_after_inclusion_dx_date"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_AnySTROKE"] = (pd.to_datetime(study_pop["AnySTROKE_after_inclusion_dx_date"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days

study_pop["PrimaryHF_after_inclusion_dx_date"] = study_pop["PrimaryHF_after_inclusion_dx_date"].fillna(study_pop["censor_date"])
study_pop["PrimaryAMI_after_inclusion_dx_date"] = pd.to_datetime(study_pop["PrimaryAMI_after_inclusion_dx_date"]).fillna(study_pop["censor_date"])
study_pop["PrimarySTROKE_after_inclusion_dx_date"] = pd.to_datetime(study_pop["PrimarySTROKE_after_inclusion_dx_date"]).fillna(study_pop["censor_date"])

study_pop["time_to_PrimaryHF"] = (pd.to_datetime(study_pop["PrimaryHF_after_inclusion_dx_date"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_PrimaryAMI"] = (pd.to_datetime(study_pop["PrimaryAMI_after_inclusion_dx_date"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_PrimarySTROKE"] = (pd.to_datetime(study_pop["PrimarySTROKE_after_inclusion_dx_date"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days

# %%
# Create composite
study_pop["AnyMACE2"] = np.where((study_pop["death"] == 1) | (study_pop["AnyAMI_after_inclusion"] == 1), 1, 0)
study_pop["AnyMACE3"] = np.where((study_pop["death"] == 1) | (study_pop["AnyAMI_after_inclusion"] == 1) | (study_pop["AnySTROKE_after_inclusion"] == 1), 1, 0)
study_pop["AnyMACE4"] = np.where((study_pop["death"] == 1) | (study_pop["AnyAMI_after_inclusion"] == 1) | (study_pop["AnySTROKE_after_inclusion"] == 1) | (study_pop["AnyHF_after_inclusion"] == 1), 1, 0)

# Create time to MACE - if 1, take the minimum of time_to_censor, time_to_HF, time_to_ami, time_to_stroke, else take time_to_censor
study_pop["time_to_AnyMACE2"] = np.where(study_pop["AnyMACE2"] == 1, study_pop[["time_to_censor", "time_to_AnyAMI"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_AnyMACE3"] = np.where(study_pop["AnyMACE3"] == 1, study_pop[["time_to_censor", "time_to_AnyAMI", "time_to_AnySTROKE"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_AnyMACE4"] = np.where(study_pop["AnyMACE4"] == 1, study_pop[["time_to_censor", "time_to_AnyHF", "time_to_AnyAMI", "time_to_AnySTROKE"]].min(axis=1), study_pop["time_to_censor"])

# Create composite
study_pop["PrimaryMACE2"] = np.where((study_pop["death"] == 1) | (study_pop["PrimaryAMI_after_inclusion"] == 1), 1, 0)
study_pop["PrimaryMACE3"] = np.where((study_pop["death"] == 1) | (study_pop["PrimaryAMI_after_inclusion"] == 1) | (study_pop["PrimarySTROKE_after_inclusion"] == 1), 1, 0)
study_pop["PrimaryMACE4"] = np.where((study_pop["death"] == 1) | (study_pop["PrimaryAMI_after_inclusion"] == 1) | (study_pop["PrimarySTROKE_after_inclusion"] == 1) | (study_pop["PrimaryHF_after_inclusion"] == 1), 1, 0)

# Create time to MACE - if 1, take the minimum of time_to_censor, time_to_HF, time_to_ami, time_to_stroke, else take time_to_censor
study_pop["time_to_PrimaryMACE2"] = np.where(study_pop["PrimaryMACE2"] == 1, study_pop[["time_to_censor", "time_to_PrimaryAMI"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_PrimaryMACE3"] = np.where(study_pop["PrimaryMACE3"] == 1, study_pop[["time_to_censor", "time_to_PrimaryAMI", "time_to_PrimarySTROKE"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_PrimaryMACE4"] = np.where(study_pop["PrimaryMACE4"] == 1, study_pop[["time_to_censor", "time_to_PrimaryHF", "time_to_PrimaryAMI", "time_to_PrimarySTROKE"]].min(axis=1), study_pop["time_to_censor"])



#%%
# To create a flag for LVSD or LVDD before ECG:
study_pop_backup_before_echo_stuff = study_pop.copy()

echo = pd.read_csv('/path_to_echo/ynhhs_echo.csv')

echo = echo.merge(study_pop[['MRN', 'ECGDate']], on = 'MRN', how = 'left')

echo['cleanedLVDiastolicFunction'] = echo['LVDiastolicFunction'].replace({'TDI Abnormal': np.nan, 'TDI Normal': 'Normal',
        'Abnormal Indeterminable': 'Indeterminate', 'indeterminate': 'Indeterminate', 'b': np.nan, 'unable': np.nan,
        'Indeterminant': 'Indeterminate', 'indetereminate': 'Indeterminate', 'Mild with elevated LA filling pressures': 'Mild',
        'Moderate to severe': 'Moderate', 'Difficult to determine due to heart block.': 'Indeterminate', 'INDETERMINATE': 'Indeterminate',
        'indeterminate ': 'Indeterminate', 'Indeterminant ': 'Indeterminate', 'Not evaluated': np.nan, 'Grade 2': 'Moderate',
        'Indeterminent': 'Indeterminate', 'Can not be determined ': 'Indeterminate',
        'There is a suggestion of inferolateral hypokinesis in some views.  Th  Mild': np.nan})

echo['ECGDate'] = pd.to_datetime(echo['ECGDate'])
echo['EchoDate'] = pd.to_datetime(echo['EchoDate'])

echo_before_ecg = echo[(pd.isna(echo['ECGDate'])) | (echo['ECGDate'] > echo['EchoDate'])]

echo_before_ecg['AnyEchoLVDD_Before_ECG'] = np.where((echo_before_ecg['cleanedLVDiastolicFunction']=='Moderate') | (echo_before_ecg['cleanedLVDiastolicFunction']=='Severe') , True, False)

echo_before_ecg['AnyEchoUnder50_Before_ECG'] = np.where(echo_before_ecg['EF']<50 , True, False)
echo_before_ecg['AnyEchoUnder40_Before_ECG'] = np.where(echo_before_ecg['EF']<40 , True, False)

echo_before_ecg['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG'] = np.where((echo_before_ecg['AnyEchoLVDD_Before_ECG']== True) | (echo_before_ecg['AnyEchoUnder50_Before_ECG']== True) , True, False)

# Group by MRN and pick up the ones that have True.

agg_dict = {
    "AnyEchoLVDD_Before_ECG": 'max',
    "AnyEchoUnder50_Before_ECG": 'max',
    "AnyEchoUnder40_Before_ECG": 'max',
    "AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG": 'max'
}

echo_before_ecg_aggregated = echo_before_ecg.groupby(['MRN']).agg(agg_dict).reset_index()

study_pop = study_pop.merge(echo_before_ecg_aggregated, how = 'left', on = 'MRN')


#%%
# To create an outcome for Future Under40, Under50, "HF hosp or Under40", "HF hosp or Under50"

echo_after_ecg = echo[(pd.isna(echo['ECGDate'])) | (echo['ECGDate'] < echo['EchoDate'])]

echo_after_ecg['AnyEchoLVDD_After_ECG'] = np.where((echo_after_ecg['cleanedLVDiastolicFunction']=='Moderate') | (echo_after_ecg['cleanedLVDiastolicFunction']=='Severe') , True, False)

echo_after_ecg['AnyEchoUnder50_After_ECG'] = np.where(echo_after_ecg['EF']<50 , True, False)
echo_after_ecg['AnyEchoUnder40_After_ECG'] = np.where(echo_after_ecg['EF']<40 , True, False)

echo_after_ecg['AnyEchoLVDD_or_AnyEchoUnder50_After_ECG'] = np.where((echo_after_ecg['AnyEchoLVDD_After_ECG']== True) | (echo_after_ecg['AnyEchoUnder50_After_ECG']== True) , True, False)


echo_after_ecg_sorted = echo_after_ecg.sort_values(by=['MRN', 'EchoDate'])

# Then, define your aggregation dictionary
agg_dict = {
    "AnyEchoLVDD_After_ECG": 'max',
    "AnyEchoUnder50_After_ECG": 'max',
    "AnyEchoUnder40_After_ECG": 'max',
    "AnyEchoLVDD_or_AnyEchoUnder50_After_ECG": 'max', 
}

# Group by 'PAT_MRN_ID' and aggregate
echo_after_ecg_aggregated = echo_after_ecg_sorted.groupby(['MRN']).agg(agg_dict).reset_index()

for echo_cond in list(agg_dict.keys()):
    first_date_df = echo_after_ecg_sorted[echo_after_ecg_sorted[echo_cond] == 1].groupby('MRN')['EchoDate'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'EchoDate': f'{echo_cond}_EchoDate'})
    echo_after_ecg_aggregated = pd.merge(echo_after_ecg_aggregated, first_date_df, on='MRN', how='left')


study_pop = study_pop.merge(echo_after_ecg_aggregated, on = 'MRN', how = 'left')

for col in ['AnyEchoLVDD_Before_ECG', 'AnyEchoUnder50_Before_ECG', 'AnyEchoUnder40_Before_ECG', 'AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG',
            'AnyEchoLVDD_After_ECG', 'AnyEchoUnder50_After_ECG', 'AnyEchoUnder40_After_ECG', 'AnyEchoLVDD_or_AnyEchoUnder50_After_ECG',]:
    study_pop[col] = study_pop[col].fillna(False)

# To create time to outcome flags for these outcomes

study_pop["AnyEchoLVDD_After_ECG_EchoDate"] = study_pop["AnyEchoLVDD_After_ECG_EchoDate"].fillna(study_pop["censor_date"])
study_pop["AnyEchoUnder50_After_ECG_EchoDate"] = pd.to_datetime(study_pop["AnyEchoUnder50_After_ECG_EchoDate"]).fillna(study_pop["censor_date"])
study_pop["AnyEchoUnder40_After_ECG_EchoDate"] = pd.to_datetime(study_pop["AnyEchoUnder40_After_ECG_EchoDate"]).fillna(study_pop["censor_date"])
study_pop["AnyEchoLVDD_or_AnyEchoUnder50_After_ECG_EchoDate"] = pd.to_datetime(study_pop["AnyEchoLVDD_or_AnyEchoUnder50_After_ECG_EchoDate"]).fillna(study_pop["censor_date"])


study_pop["time_to_AnyEchoLVDD_After_ECG"] = (pd.to_datetime(study_pop["AnyEchoLVDD_After_ECG_EchoDate"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_AnyEchoUnder50_After_ECG"] = (pd.to_datetime(study_pop["AnyEchoUnder50_After_ECG_EchoDate"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_AnyEchoUnder40_After_ECG"] = (pd.to_datetime(study_pop["AnyEchoUnder40_After_ECG_EchoDate"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days
study_pop["time_to_AnyEchoLVDD_or_AnyEchoUnder50_After_ECG"] = (pd.to_datetime(study_pop["AnyEchoLVDD_or_AnyEchoUnder50_After_ECG_EchoDate"]) - pd.to_datetime(study_pop["ECGDate"])).dt.days

#%%
# Create the aggregate outcome of "(Echo Under 50 or HF Hosp) etc"
    
study_pop["PrimaryHF_or_EchoUnder50"] = np.where((study_pop["AnyEchoUnder50_After_ECG"] == 1) | (study_pop["PrimaryHF_after_inclusion"] == 1), 1, 0)
study_pop["PrimaryHF_or_EchoUnder40"] = np.where((study_pop["AnyEchoUnder40_After_ECG"] == 1) | (study_pop["PrimaryHF_after_inclusion"] == 1), 1, 0)
study_pop["PrimaryHF_or_EchoLVDD_or_EchoUnder50"] = np.where((study_pop["AnyEchoLVDD_or_AnyEchoUnder50_After_ECG"] == 1) | (study_pop["PrimaryHF_after_inclusion"] == 1), 1, 0)

# Create time to MACE - if 1, take the minimum of time_to_censor, time_to_HF, time_to_ami, time_to_stroke, else take time_to_censor
study_pop["time_to_PrimaryHF_or_EchoUnder50"] = np.where(study_pop["PrimaryHF_or_EchoUnder50"] == 1, study_pop[["time_to_AnyEchoUnder50_After_ECG", "time_to_PrimaryHF"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_PrimaryHF_or_EchoUnder40"] = np.where(study_pop["PrimaryHF_or_EchoUnder40"] == 1, study_pop[["time_to_AnyEchoUnder40_After_ECG", "time_to_PrimaryHF"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_PrimaryHF_or_EchoLVDD_or_EchoUnder50"] = np.where(study_pop["PrimaryHF_or_EchoLVDD_or_EchoUnder50"] == 1, study_pop[["time_to_AnyEchoLVDD_or_AnyEchoUnder50_After_ECG", "time_to_PrimaryHF",]].min(axis=1), study_pop["time_to_censor"])


#%%
study_pop["AnyHF_or_EchoUnder50"] = np.where((study_pop["AnyEchoUnder50_After_ECG"] == 1) | (study_pop["AnyHF_after_inclusion"] == 1), 1, 0)
study_pop["AnyHF_or_EchoUnder40"] = np.where((study_pop["AnyEchoUnder40_After_ECG"] == 1) | (study_pop["AnyHF_after_inclusion"] == 1), 1, 0)
study_pop["AnyHF_or_EchoLVDD_or_EchoUnder50"] = np.where((study_pop["AnyEchoLVDD_or_AnyEchoUnder50_After_ECG"] == 1) | (study_pop["AnyHF_after_inclusion"] == 1), 1, 0)

# Create time to MACE - if 1, take the minimum of time_to_censor, time_to_HF, time_to_ami, time_to_stroke, else take time_to_censor
study_pop["time_to_AnyHF_or_EchoUnder50"] = np.where(study_pop["AnyHF_or_EchoUnder50"] == 1, study_pop[["time_to_AnyEchoUnder50_After_ECG", "time_to_AnyHF"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_AnyHF_or_EchoUnder40"] = np.where(study_pop["AnyHF_or_EchoUnder40"] == 1, study_pop[["time_to_AnyEchoUnder40_After_ECG", "time_to_AnyHF"]].min(axis=1), study_pop["time_to_censor"])
study_pop["time_to_AnyHF_or_EchoLVDD_or_EchoUnder50"] = np.where(study_pop["AnyHF_or_EchoLVDD_or_EchoUnder50"] == 1, study_pop[["time_to_AnyEchoLVDD_or_AnyEchoUnder50_After_ECG", "time_to_AnyHF",]].min(axis=1), study_pop["time_to_censor"])

#%%
# Check a few counts
print('People without AnyHF before ECGDate, but PrimaryHF after ECGDate: ', len(study_pop[(study_pop['AnyHF_before_inclusion']==0) & (study_pop['PrimaryHF_after_inclusion']==1)]))

print('People without AnyHF or Under50 or LVDD before ECGDate, but PrimaryHF after ECGDate: ', len(study_pop[(study_pop['AnyHF_before_inclusion']==0) & (study_pop['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG'] == False) & (study_pop['PrimaryHF_after_inclusion']==1)]))
print('People without AnyHF or Under50 or LVDD before ECGDate, but PrimaryHF or EchoUnder40 after ECGDate: ', len(study_pop[(study_pop['AnyHF_before_inclusion']==0) & (study_pop['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG'] == False) & (study_pop['PrimaryHF_or_EchoUnder40']==1)]))
print('People without AnyHF or Under50 or LVDD before ECGDate, but PrimaryHF or EchoUnder50 after ECGDate: ', len(study_pop[(study_pop['AnyHF_before_inclusion']==0) & (study_pop['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG'] == False) & (study_pop['PrimaryHF_or_EchoUnder50']==1)]))
print('People without AnyHF or Under50 or LVDD before ECGDate, but AnyHF after ECGDate: ', len(study_pop[(study_pop['AnyHF_before_inclusion']==0) & (study_pop['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG'] == False) & (study_pop['AnyHF_after_inclusion']==1)]))
print('People without AnyHF or Under50 or LVDD before ECGDate, but AnyHF or EchoUnder50 after ECGDate: ', len(study_pop[(study_pop['AnyHF_before_inclusion']==0) & (study_pop['AnyEchoLVDD_or_AnyEchoUnder50_Before_ECG'] == False) & (study_pop['AnyHF_or_EchoUnder50']==1)]))

# %%
train_df_single_lead_model = pd.read_csv('/path_to_train_df/train_df.csv', low_memory = False)
study_pop['MRN_in_trainDF_single_lead_model'] = np.where(study_pop['MRN'].isin(train_df_single_lead_model['MRN'].unique().tolist()), True, False)

#%%
# Merging with predictions

single_lead_model_preds = pd.read_csv('/path_to_ecg/ynhhs_ecgs_1_Lead_preds.csv', low_memory = False).rename(columns={'preds_Under40': 'preds_EF_single_lead_model'})
single_lead_model_preds = single_lead_model_preds[['fileID', 'preds_EF_single_lead_model']]
study_pop = study_pop.merge(single_lead_model_preds, how = 'left', on = 'fileID')

#%%
# To define ECGs during inpatient visits
hosp_enc_df_cols = ['PAT_MRN_ID', 'PAT_ENC_CSN_ID', 'HOSP_ADMSN_DATE', 'HOSP_DISCH_DATE', 'ACCT_BASECLS_HA',]

# Read hosp_enc df
hosp_enc = read_full(path, 'Hospital_Enc.txt')[hosp_enc_df_cols]


# Filter for inpatient
inpt_enc = hosp_enc[hosp_enc['ACCT_BASECLS_HA']=='Inpatient']

# Merge study_pop and inpt on MRNs - 
# Then filter for those that happended within the duration of the visit

inpt_study_pop_merge = study_pop.merge(inpt_enc, how = 'inner', on = 'PAT_MRN_ID')

for col in ['ECGDate', 'HOSP_ADMSN_DATE', 'HOSP_DISCH_DATE']:
    inpt_study_pop_merge[col] = pd.to_datetime(inpt_study_pop_merge[col])

during_inpt = inpt_study_pop_merge[(inpt_study_pop_merge['ECGDate']>=inpt_study_pop_merge['HOSP_ADMSN_DATE'])&(inpt_study_pop_merge['ECGDate']<=inpt_study_pop_merge['HOSP_DISCH_DATE'])]

study_pop['ECG_during_inpatient_visit'] = np.where(study_pop['fileID'].isin(during_inpt['fileID'].unique()), True, False)


#%%
# Removing those with high BNP
bnp_labs1 = pd.read_csv(path + 'bnp_hf_labs1.csv')
bnp_labs2 = pd.read_csv(path + 'bnp_hf_labs2.csv')
bnp_labs3 = pd.read_csv(path + 'bnp_hf_labs3.csv')

bnp = pd.concat([bnp_labs1, bnp_labs2, bnp_labs3])

bnp2 = bnp.merge(study_pop[['PAT_MRN_ID', 'ECGDate']], how = 'left', on = 'PAT_MRN_ID')
for col in ['ECGDate','LAB_DATE']:
    bnp2[col] = pd.to_datetime(bnp2[col])
bnp2['BNP_ECG_delta'] = bnp2['ECGDate'] - bnp2['LAB_DATE']

# Keep the highest BNP value before the ECG: 

bnp2 = bnp2[bnp2['BNP_ECG_delta'].dt.days > 0]
bnp2 = bnp2.sort_values(by = ['ORD_NUM_VALUE_CALC'], ascending = False).drop_duplicates(subset = ['PAT_MRN_ID'], keep = 'first')

# Rename ORD_NUM_VALUE_CALC to Highest_BNP_before_ECG

bnp2 = bnp2.rename(columns = {'ORD_NUM_VALUE_CALC': 'Highest_BNP_before_ECG'})
bnp2['Highest_preECG_BNP_time_delta'] = bnp2['BNP_ECG_delta'].dt.days

# Merge with study_pop
study_pop = study_pop.merge(bnp2[['PAT_MRN_ID', 'Highest_BNP_before_ECG', 'Highest_preECG_BNP_time_delta']], how = 'left', on = 'PAT_MRN_ID')

# Export study_pop
study_pop.to_csv('/path_to_yale/yale_df_processed.csv') 
#%%