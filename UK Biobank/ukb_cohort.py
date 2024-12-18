# %%
import pandas as pd
import numpy as np
# import datatable as dt
import math
from datetime import date
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from Equations.pcp_hf_equation import *
from Equations.prevent_hf_equation import *


ukb_ecg = pd.read_csv('/path_to_ukb/ukb_ecgs_info.csv')

#This dataset has all the data with relevant columns (2K instead of 15K) to this project
ukb_data = pd.read_csv("/path_to_ukb/filtered_ukb47034.csv")

ukb_data = ukb_data[ukb_data['eid'].isin(ukb_ecg['eid'].to_list())]
ukb_key = pd.read_excel("/path_to_ukb/my_ukb_key.xlsx")

# %%
#Demographics
demo_ukb_data = ukb_data[['eid','sex_f31_0_0','year_of_birth_f34_0_0','month_of_birth_f52_0_0',
                          'date_of_attending_assessment_centre_f53_2_0','date_of_attending_assessment_centre_f53_3_0',
                          'ethnic_background_f21000_0_0','ethnic_background_f21000_1_0',
                          'ethnic_background_f21000_2_0', 'date_lost_to_followup_f191_0_0', 'date_of_death_f40000_0_0', 'date_of_death_f40000_1_0']]

demo_ukb_data['birth_date'] = pd.to_datetime((demo_ukb_data[demo_ukb_data['year_of_birth_f34_0_0'].notna()][
                                                'year_of_birth_f34_0_0'].astype(int).astype(str) + '_' + 
                                             demo_ukb_data[demo_ukb_data['month_of_birth_f52_0_0'].notna()][
                                                'month_of_birth_f52_0_0'].astype(int).astype(str) + '_' + '15'), format='%Y_%m_%d')
demo_ukb_data['ethnicity'] = np.where(demo_ukb_data['ethnic_background_f21000_0_0'].notna(), 
                                      demo_ukb_data['ethnic_background_f21000_0_0'], 
                                      np.where(demo_ukb_data['ethnic_background_f21000_1_0'].notna(), 
                                      demo_ukb_data['ethnic_background_f21000_1_0'], 
                                      np.where(demo_ukb_data['ethnic_background_f21000_2_0'].notna(),
                                      demo_ukb_data['ethnic_background_f21000_2_0'], np.nan)))

demo_ukb_data = demo_ukb_data.rename(columns={'sex_f31_0_0':'sex',
                                              'date_of_attending_assessment_centre_f53_2_0':'visit_2',
                                              'date_of_attending_assessment_centre_f53_3_0':'visit_3',
                                              'date_lost_to_followup_f191_0_0':'lost_fu_date'}).drop(columns = 
                                              ['year_of_birth_f34_0_0','month_of_birth_f52_0_0','ethnic_background_f21000_0_0',
                                              'ethnic_background_f21000_1_0','ethnic_background_f21000_2_0'])

demo_ukb_data['death_date'] = np.where(demo_ukb_data['date_of_death_f40000_0_0'].notna(), demo_ukb_data['date_of_death_f40000_0_0'],
                                            np.where(demo_ukb_data['date_of_death_f40000_1_0'].notna(), demo_ukb_data['date_of_death_f40000_1_0'], np.nan))

demo_ukb_data = demo_ukb_data[['eid', 'birth_date', 'sex', 'ethnicity',
                               'visit_2', 'visit_3', 'lost_fu_date', 'death_date']]

for i in ['birth_date', 'visit_2', 'visit_3', 'lost_fu_date', 'death_date']:
    demo_ukb_data[i] = pd.to_datetime(demo_ukb_data[i])

demo_ukb_data['age_2'] = (demo_ukb_data['visit_2'] - demo_ukb_data['birth_date']
                        ).astype(str).str.split(" ", expand=True)[0].replace('NaT', np.nan).astype(float).mul(1/365.25)
demo_ukb_data['age_3'] = pd.to_numeric((demo_ukb_data['visit_3'] - demo_ukb_data['birth_date']
                          ).astype(str).str.split(" ", expand=True)[0], errors='coerce').astype(float).mul(1/365.25)


for i in ['_2', '_3']:
    demo_ukb_data['Death' + i] = np.where((demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']>=demo_ukb_data['visit' + i]), 1, np.where(
        (demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']<demo_ukb_data['visit' + i]), np.nan, 0))
    demo_ukb_data['Time2Death' + i] = np.where(demo_ukb_data['Death' + i].isna(), np.nan, np.where(
                                demo_ukb_data['Death' + i]==1, (demo_ukb_data["death_date"] - demo_ukb_data['visit' + i]).dt.days,
                                np.where(demo_ukb_data["lost_fu_date"].notna(), (demo_ukb_data["lost_fu_date"] - demo_ukb_data['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - demo_ukb_data['visit' + i]).dt.days)))

for i in ['_2', '_3']:
    demo_ukb_data['Death' + i] = np.where((demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']>=demo_ukb_data['visit' + i]), 1, np.where(
        (demo_ukb_data['death_date'].notna()) & (demo_ukb_data['death_date']<demo_ukb_data['visit' + i]), np.nan, 0))
    demo_ukb_data['Time2Death' + i] = np.where(demo_ukb_data['Death' + i].isna(), np.nan, np.where(
                                demo_ukb_data['Death' + i]==1, (demo_ukb_data["death_date"] - demo_ukb_data['visit' + i]).dt.days,
                                np.where((demo_ukb_data["lost_fu_date"].notna()) & (demo_ukb_data['lost_fu_date']>=demo_ukb_data['visit' + i]),
                                            (demo_ukb_data["lost_fu_date"] - demo_ukb_data['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - demo_ukb_data['visit' + i]).dt.days)))
    
for i in ['Death', 'Time2Death']:
    demo_ukb_data[i + '_2'] = np.where((demo_ukb_data['visit_2'].isna()) | (demo_ukb_data['lost_fu_date']<demo_ukb_data['visit_2']), np.nan, demo_ukb_data[i + '_2'])
    demo_ukb_data[i + '_3'] = np.where((demo_ukb_data['visit_3'].isna()) | (demo_ukb_data['lost_fu_date']<demo_ukb_data['visit_3']), np.nan, demo_ukb_data[i + '_3'])


#sex_coding = dict({0:'Female',1:'Male'})

ethnicity = dict({1:1,1001:1,1002:1,1003:1,
                  2:2,2001:2,2002:2,2003:2,2004:2,
                  3:3,3001:3,3002:3,3003:3,3004:3,
                  4:4,4001:4,4002:4,4003:4,
                  5:5,
                  6:6,
                  -1:np.nan, -3:np.nan})
demo_ukb_data = demo_ukb_data.replace({'ethnicity': ethnicity})

#ethnicity_cat = dict({1:'White',2:"Mixed",3:'South Asian',4:'Black',5:'Chinese',6:"Other"})

demo_ukb_data['race_sex_group'] = np.where((demo_ukb_data['sex']==0) & (demo_ukb_data['sex']==1), 'White Female', np.where(
    (demo_ukb_data['sex']==1) & (demo_ukb_data['sex']==1), 'White Male', np.where(
        (demo_ukb_data['sex']==0) & (demo_ukb_data['sex']==4), 'Black Female', np.where(
            (demo_ukb_data['sex']==1) & (demo_ukb_data['sex']==4), 'Black Male', 'Others'))))
demo_ukb_data

# %%
#Cause of death
death_cause = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([40001, 40002])]['col.name'].tolist()]
primary_death_stubname = ukb_key[ukb_key['field.showcase'] == 40001]['col.name'].iloc[0].rsplit('_', 2)[0]
secondary_death_stubname = ukb_key[ukb_key['field.showcase'] == 40002]['col.name'].iloc[0].rsplit('_', 2)[0]
death_cause = pd.wide_to_long(df=death_cause, stubnames=[primary_death_stubname, secondary_death_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
death_cause[['instance','array']] = death_cause['instance_array'].str.split("_", expand=True)
death_cause.rename(columns={'underlying_primary_cause_of_death_icd10_f40001':'primary_cause_death',
                            'contributory_secondary_ca_f40002':'secondary_cause_death'},inplace=True)

#Defining cause of death

hf_death_list = [
#heart_failure_list
            'I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
]

death_cause['primary_hf_mortality'] = np.where(death_cause['primary_cause_death'].isna(), np.nan,
                                               np.where(death_cause['primary_cause_death'].isin(hf_death_list), 1, 0))
death_cause['secondary_hf_mortality'] = np.where(death_cause['secondary_cause_death'].isna(), np.nan,
                                               np.where(death_cause['secondary_cause_death'].isin(hf_death_list), 1, 0))

death_cause['primary_hf_mortality'] = death_cause.groupby('eid')['primary_hf_mortality'].transform(max)
death_cause['secondary_hf_mortality'] = death_cause.groupby('eid')['secondary_hf_mortality'].transform(max)
death_cause.drop_duplicates(subset='eid', inplace=True)
death_cause = death_cause[['eid', 'primary_hf_mortality', 'secondary_hf_mortality']]
death_cause['primary_secondary_hf_mortality'] = np.where((death_cause['primary_hf_mortality']==1)|(death_cause['secondary_hf_mortality']==1), 1,
                                                         np.where((death_cause['primary_hf_mortality']==0)|(death_cause['secondary_hf_mortality']==0), 0, np.nan))
demo_ukb_data = demo_ukb_data.merge(death_cause, on='eid', how='left')
demo_ukb_data 

# %%
#MACE, primary diagnosis
#ICD10
pmh_ukb_maindx_icd10 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41202,41262])]['col.name'].tolist()]
icd10_maindx_stubname = ukb_key[ukb_key['field.showcase'] == 41202]['col.name'].iloc[0].rsplit('_', 2)[0]
icd10_maindate_stubname = ukb_key[ukb_key['field.showcase'] == 41262]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_maindx_icd10 = pd.wide_to_long(df=pmh_ukb_maindx_icd10, stubnames=[icd10_maindx_stubname, icd10_maindate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_maindx_icd10 = long_pmh_ukb_maindx_icd10.drop(columns='instance_array').rename(
    columns={'diagnoses_main_icd10_f41202':'icd', 'date_of_first_inpatient_diagnosis_main_icd10_f41262':'main_icd_date'})
final_pmh_ukb_maindx_icd10['icd_type'] = 'icd10'

#ICD9
pmh_ukb_maindx_icd9 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41203,41263])]['col.name'].tolist()]
icd9_maindx_stubname = ukb_key[ukb_key['field.showcase'] == 41203]['col.name'].iloc[0].rsplit('_', 2)[0]
icd9_maindate_stubname = ukb_key[ukb_key['field.showcase'] == 41263]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_maindx_icd9 = pd.wide_to_long(df=pmh_ukb_maindx_icd9, stubnames=[icd9_maindx_stubname, icd9_maindate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_maindx_icd9 = long_pmh_ukb_maindx_icd9.drop(columns='instance_array').rename(columns={
    'diagnoses_main_icd9_f41203':'icd', 'date_of_first_inpatient_diagnosis_main_icd9_f41263':'main_icd_date'})
final_pmh_ukb_maindx_icd9['icd_type'] = 'icd9'

#Merging ICD9 and ICD10
pmh_ukb_maindx = pd.concat([final_pmh_ukb_maindx_icd10, final_pmh_ukb_maindx_icd9], ignore_index=True)
pmh_ukb_maindx = pmh_ukb_maindx.merge(demo_ukb_data, on='eid', how='left')
pmh_ukb_maindx['main_icd_date'] = pd.to_datetime(pmh_ukb_maindx['main_icd_date'])

#Creating variables for episodes occurred at or before visit_0 and visit_1
#Defining Each Disease Using ICD-9/ICD-10 Codes
ihd_hosp = ['I200', 'I21', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219',
            'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I24', 'I248',
            'I249',

            '410', '4109']

stroke_hosp=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459',
             'I63', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638',
             'I639', 'I64',
             
             
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434',
             '4340', '4341', '4349', '435', '4359']

heart_failure_hosp = ['I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
                     
                      '428','4280','4281','4289']

pmh_hosp = {'ihd': ihd_hosp, 'stroke': stroke_hosp, 'heart_failure': heart_failure_hosp}
hosp_list = ['ihd', 'stroke', 'heart_failure'] 

pmh_ukb_maindx['ihd'] = np.where(pmh_ukb_maindx['icd'].isin(ihd_hosp), 1, 0)
pmh_ukb_maindx['ihd_date'] = np.where(pmh_ukb_maindx['ihd']==1, pmh_ukb_maindx['main_icd_date'], pd.NaT)
pmh_ukb_maindx['stroke'] = np.where(pmh_ukb_maindx['icd'].isin(stroke_hosp), 1, 0)
pmh_ukb_maindx['stroke_date'] = np.where(pmh_ukb_maindx['stroke']==1, pmh_ukb_maindx['main_icd_date'], pd.NaT)
pmh_ukb_maindx['heart_failure'] = np.where(pmh_ukb_maindx['icd'].isin(heart_failure_hosp), 1, 0)
pmh_ukb_maindx['hf_date'] = np.where(pmh_ukb_maindx['heart_failure']==1, pmh_ukb_maindx['main_icd_date'], pd.NaT)

# Defining earliest heart failure date
pmh_ukb_maindx['earliest_date_hf_primary'] = pmh_ukb_maindx.groupby('eid')['hf_date'].transform(min)

for i in ['ihd_date', 'stroke_date', 'hf_date', 'earliest_date_hf_primary']:
    pmh_ukb_maindx[i] = pd.to_datetime(pmh_ukb_maindx[i])

for i in ['_2', '_3']:
    pmh_ukb_maindx['IHD' + i] = np.where((pmh_ukb_maindx['ihd_date'].notna()) & (pmh_ukb_maindx['ihd_date']>=pmh_ukb_maindx['visit' + i]), 1, np.where(
        (pmh_ukb_maindx['ihd_date'].notna()) & (pmh_ukb_maindx['ihd_date']<pmh_ukb_maindx['visit' + i]), np.nan, 0))
    pmh_ukb_maindx['Time2IHD' + i] = np.where(pmh_ukb_maindx['IHD' + i].isna(), np.nan, np.where(
                                pmh_ukb_maindx['IHD' + i]==1, (pmh_ukb_maindx["ihd_date"] - pmh_ukb_maindx['visit' + i]).dt.days,
                                np.where((pmh_ukb_maindx["lost_fu_date"].notna()) & (pmh_ukb_maindx['lost_fu_date']>=pmh_ukb_maindx['visit' + i]),
                                            (pmh_ukb_maindx["lost_fu_date"] - pmh_ukb_maindx['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_maindx['visit' + i]).dt.days)))
    
    pmh_ukb_maindx['Stroke' + i] = np.where((pmh_ukb_maindx['stroke_date'].notna()) & (pmh_ukb_maindx['stroke_date']>=pmh_ukb_maindx['visit' + i]), 1, np.where(
        (pmh_ukb_maindx['stroke_date'].notna()) & (pmh_ukb_maindx['stroke_date']<pmh_ukb_maindx['visit' + i]), np.nan, 0))
    pmh_ukb_maindx['Time2Stroke' + i] = np.where(pmh_ukb_maindx['Stroke' + i].isna(), np.nan, np.where(
                                pmh_ukb_maindx['Stroke' + i]==1, (pmh_ukb_maindx["stroke_date"] - pmh_ukb_maindx['visit' + i]).dt.days,
                                np.where((pmh_ukb_maindx["lost_fu_date"].notna()) & (pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['visit' + i]),
                                            (pmh_ukb_maindx["lost_fu_date"] - pmh_ukb_maindx['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_maindx['visit' + i]).dt.days)))

    pmh_ukb_maindx['HF' + i] = np.where((pmh_ukb_maindx['hf_date'].notna()) & (pmh_ukb_maindx['hf_date']>=pmh_ukb_maindx['visit' + i]), 1, np.where(
        (pmh_ukb_maindx['hf_date'].notna()) & (pmh_ukb_maindx['hf_date']<pmh_ukb_maindx['visit' + i]), np.nan, 0))
    pmh_ukb_maindx['Time2HF' + i] = np.where(pmh_ukb_maindx['HF' + i].isna(), np.nan, np.where(
                                pmh_ukb_maindx['HF' + i]==1, (pmh_ukb_maindx["hf_date"] - pmh_ukb_maindx['visit' + i]).dt.days,
                                np.where((pmh_ukb_maindx["lost_fu_date"].notna()) & (pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['visit' + i]),
                                            (pmh_ukb_maindx["lost_fu_date"] - pmh_ukb_maindx['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_maindx['visit' + i]).dt.days)))
    
for i in ['IHD', 'Time2IHD', 'Stroke', 'Time2Stroke', 'HF', 'Time2HF']:
    pmh_ukb_maindx[i + '_2'] = np.where((pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['visit_2']) | (pmh_ukb_maindx['visit_2'].isna()), 
                                        np.nan, pmh_ukb_maindx[i + '_2'])
    pmh_ukb_maindx[i + '_3'] = np.where((pmh_ukb_maindx['lost_fu_date']<pmh_ukb_maindx['visit_3']) | (pmh_ukb_maindx['visit_3'].isna()), 
                                        np.nan, pmh_ukb_maindx[i + '_3'])

for i in ['IHD_2', 'IHD_3', 'Stroke_2', 'Stroke_3', 'HF_2', 'HF_3']:
    pmh_ukb_maindx[i] = pmh_ukb_maindx.groupby('eid')[i].transform(max)

for i in ['IHD_2', 'IHD_3', 'Stroke_2', 'Stroke_3', 'HF_2', 'HF_3']:
    pmh_ukb_maindx['Time2' + i] = pmh_ukb_maindx.groupby('eid')['Time2' + i].transform(min)

pmh_ukb_maindx.drop_duplicates(subset='eid', inplace=True)

demo_ukb_data = pmh_ukb_maindx

for i in ['_2', '_3']:
    demo_ukb_data['MACE' + i] = np.where((demo_ukb_data['IHD' + i]==1) | (demo_ukb_data['Stroke' + i]==1) | (demo_ukb_data['HF' + i]==1) | (demo_ukb_data['Death' + i]==1), 1,
                            np.where((demo_ukb_data['IHD' + i]==0) & (demo_ukb_data['Stroke' + i]==0) & (demo_ukb_data['HF' + i]==0) & (demo_ukb_data['Death' + i]==0), 0, np.nan))
    demo_ukb_data['Time2MACE' + i] = np.where(demo_ukb_data['MACE' + i]==1, demo_ukb_data[['Time2IHD' + i, 'Time2Stroke' + i, 'Time2HF' + i, 'Time2Death' + i]].min(axis=1),
                        np.where(demo_ukb_data['MACE' + i]==0, demo_ukb_data[['Time2IHD' + i, 'Time2Stroke' + i,  'Time2HF' + i, 'Time2Death' + i]].max(axis=1), np.nan))    
    
demo_ukb_data.drop(columns=['icd', 'main_icd_date', 'icd_type', 'ihd', 'ihd_date', 'stroke', 'stroke_date', 'heart_failure', 'hf_date'], inplace=True)
for column in [col for col in demo_ukb_data.columns[demo_ukb_data.columns.str.contains('_2|_3')] if not col in ['visit_2', 'visit_3', 'age_2', 'age_3', 'Death_2', 'Time2Death_2', 'Death_3', 'Time2Death_3']]:
    demo_ukb_data.rename(columns={column: 'primary_' + column}, inplace=True)
demo_ukb_data

# %%
#MACE, All diagnosis
#ICD10
pmh_ukb_alldx_icd10 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41270,41280])]['col.name'].tolist()]
icd10_alldx_stubname = ukb_key[ukb_key['field.showcase'] == 41270]['col.name'].iloc[0].rsplit('_', 2)[0]
icd10_alldate_stubname = ukb_key[ukb_key['field.showcase'] == 41280]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_alldx_icd10 = pd.wide_to_long(df=pmh_ukb_alldx_icd10, stubnames=[icd10_alldx_stubname, icd10_alldate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_alldx_icd10 = long_pmh_ukb_alldx_icd10.drop(columns='instance_array').rename(
    columns={'diagnoses_icd10_f41270':'icd', 'date_of_first_inpatient_diagnosis_icd10_f41280':'all_icd_date'})
final_pmh_ukb_alldx_icd10['icd_type'] = 'icd10'

#ICD9
pmh_ukb_alldx_icd9 = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([41271,41281])]['col.name'].tolist()]
icd9_alldx_stubname = ukb_key[ukb_key['field.showcase'] == 41271]['col.name'].iloc[0].rsplit('_', 2)[0]
icd9_alldate_stubname = ukb_key[ukb_key['field.showcase'] == 41281]['col.name'].iloc[0].rsplit('_', 2)[0]

long_pmh_ukb_alldx_icd9 = pd.wide_to_long(df=pmh_ukb_alldx_icd9, stubnames=[icd9_alldx_stubname, icd9_alldate_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
final_pmh_ukb_alldx_icd9 = long_pmh_ukb_alldx_icd9.drop(columns='instance_array').rename(columns={
    'diagnoses_icd9_f41271':'icd', 'date_of_first_inpatient_diagnosis_icd9_f41281':'all_icd_date'})
final_pmh_ukb_alldx_icd9['icd_type'] = 'icd9'

#Merging ICD9 and ICD10
pmh_ukb_alldx = pd.concat([final_pmh_ukb_alldx_icd10, final_pmh_ukb_alldx_icd9], ignore_index=True)
pmh_ukb_alldx = pmh_ukb_alldx.merge(demo_ukb_data, on='eid', how='left')
pmh_ukb_alldx['all_icd_date'] = pd.to_datetime(pmh_ukb_alldx['all_icd_date'])

#Creating variables for episodes occurred at or before visit_0 and visit_1
#Defining Each Disease Using ICD-9/ICD-10 Codes
ihd_hosp = ['I200', 'I21', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219',
            'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I24', 'I248',
            'I249',

            '410', '4109']

stroke_hosp=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459',
             'I63', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638',
             'I639', 'I64',
             
             
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434',
             '4340', '4341', '4349', '435', '4359']

heart_failure_hosp = ['I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
                     
                      '428','4280','4281','4289']

htn_list = ['I10', 'I11', 'I110', 'I119', 'I12', 'I120', 'I129', 'I13', 'I130', 'I131',
            'I132', 'I139', 'I674', 'O10', 'O100', 'O101', 'O102', 'O103', 'O109', 'O11',
            
            '401', '4010', '4011', '4019', '402', '4020', '4021', '4029', '403', '4030',
            '4031', '4039', '404', '4040', '4041', '4049', '6420', '6422', '6427', '6429']
dm_list = ['E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117',
           'E118', 'E119', 'E12', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126',
           'E127', 'E128', 'E129', 'E13', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135',
           'E136', 'E137', 'E138', 'E139', 'E14', 'E140', 'E141', 'E142', 'E143', 'E144',
           'E145', 'E146', 'E147', 'E148', 'E149', 'O241', 'O242', 'O243', 'O249',
    
           '250', '2500', '25000', '25009', '2501', '25010', '25019', '2502',
           '25020', '25029', '2503', '2504', '2505', '2506', '2507', '2509', '25090',
           '25099', '6480']

ckd_list = ['I12', 'I120', 'I13', 'I130', 'I131', 'I132', 'I139', 'N18', 'N180', 'N181',
            'N182', 'N183', 'N184', 'N185', 'N188', 'N189', 'Z49', 'Z490', 'Z491', 'Z492',
            
            '403', '4030', '4031', '4039', '404', '4040', '4041', '4049', '585', '5859',
            '6421', '6462']

pmh_hosp = {'ihd': ihd_hosp, 'stroke': stroke_hosp, 'heart_failure': heart_failure_hosp,
            'htn': htn_list, 'dm': dm_list, 'ckd': ckd_list}
hosp_list = ['ihd', 'stroke', 'heart_failure', 'htn', 'dm', 'ckd'] 

pmh_ukb_alldx['ihd'] = np.where(pmh_ukb_alldx['icd'].isin(ihd_hosp), 1, 0)
pmh_ukb_alldx['ihd_date'] = np.where(pmh_ukb_alldx['ihd']==1, pmh_ukb_alldx['all_icd_date'], pd.NaT)
pmh_ukb_alldx['stroke'] = np.where(pmh_ukb_alldx['icd'].isin(stroke_hosp), 1, 0)
pmh_ukb_alldx['stroke_date'] = np.where(pmh_ukb_alldx['stroke']==1, pmh_ukb_alldx['all_icd_date'], pd.NaT)
pmh_ukb_alldx['heart_failure'] = np.where(pmh_ukb_alldx['icd'].isin(heart_failure_hosp), 1, 0)
pmh_ukb_alldx['hf_date'] = np.where(pmh_ukb_alldx['heart_failure']==1, pmh_ukb_alldx['all_icd_date'], pd.NaT)

pmh_ukb_alldx['htn'] = np.where(pmh_ukb_alldx['icd'].isin(htn_list), 1, 0)
pmh_ukb_alldx['htn_date'] = np.where(pmh_ukb_alldx['htn']==1, pmh_ukb_alldx['all_icd_date'], pd.NaT)
pmh_ukb_alldx['dm'] = np.where(pmh_ukb_alldx['icd'].isin(dm_list), 1, 0)
pmh_ukb_alldx['dm_date'] = np.where(pmh_ukb_alldx['dm']==1, pmh_ukb_alldx['all_icd_date'], pd.NaT)
pmh_ukb_alldx['ckd'] = np.where(pmh_ukb_alldx['icd'].isin(ckd_list), 1, 0)
pmh_ukb_alldx['ckd_date'] = np.where(pmh_ukb_alldx['ckd']==1, pmh_ukb_alldx['all_icd_date'], pd.NaT)

# Defining earliest heart failure date
pmh_ukb_alldx['earliest_date_hf_all'] = pmh_ukb_alldx.groupby('eid')['hf_date'].transform(min)

for i in ['ihd_date', 'stroke_date', 'hf_date', 'earliest_date_hf_all', 'htn_date', 'dm_date', 'ckd_date']:
    pmh_ukb_alldx[i] = pd.to_datetime(pmh_ukb_alldx[i])

for i in ['_2', '_3']:
    pmh_ukb_alldx['IHD' + i] = np.where((pmh_ukb_alldx['ihd_date'].notna()) & (pmh_ukb_alldx['ihd_date']>=pmh_ukb_alldx['visit' + i]), 1, np.where(
        (pmh_ukb_alldx['ihd_date'].notna()) & (pmh_ukb_alldx['ihd_date']<pmh_ukb_alldx['visit' + i]), np.nan, 0))
    pmh_ukb_alldx['Time2IHD' + i] = np.where(pmh_ukb_alldx['IHD' + i].isna(), np.nan, np.where(
                                pmh_ukb_alldx['IHD' + i]==1, (pmh_ukb_alldx["ihd_date"] - pmh_ukb_alldx['visit' + i]).dt.days,
                                np.where((pmh_ukb_alldx["lost_fu_date"].notna()) & (pmh_ukb_alldx['lost_fu_date']>=pmh_ukb_alldx['visit' + i]),
                                            (pmh_ukb_alldx["lost_fu_date"] - pmh_ukb_alldx['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_alldx['visit' + i]).dt.days)))
    
    pmh_ukb_alldx['Stroke' + i] = np.where((pmh_ukb_alldx['stroke_date'].notna()) & (pmh_ukb_alldx['stroke_date']>=pmh_ukb_alldx['visit' + i]), 1, np.where(
        (pmh_ukb_alldx['stroke_date'].notna()) & (pmh_ukb_alldx['stroke_date']<pmh_ukb_alldx['visit' + i]), np.nan, 0))
    pmh_ukb_alldx['Time2Stroke' + i] = np.where(pmh_ukb_alldx['Stroke' + i].isna(), np.nan, np.where(
                                pmh_ukb_alldx['Stroke' + i]==1, (pmh_ukb_alldx["stroke_date"] - pmh_ukb_alldx['visit' + i]).dt.days,
                                np.where((pmh_ukb_alldx["lost_fu_date"].notna()) & (pmh_ukb_alldx['lost_fu_date']<pmh_ukb_alldx['visit' + i]),
                                            (pmh_ukb_alldx["lost_fu_date"] - pmh_ukb_alldx['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_alldx['visit' + i]).dt.days)))

    pmh_ukb_alldx['HF' + i] = np.where((pmh_ukb_alldx['hf_date'].notna()) & (pmh_ukb_alldx['hf_date']>=pmh_ukb_alldx['visit' + i]), 1, np.where(
        (pmh_ukb_alldx['hf_date'].notna()) & (pmh_ukb_alldx['hf_date']<pmh_ukb_alldx['visit' + i]), np.nan, 0))
    pmh_ukb_alldx['Time2HF' + i] = np.where(pmh_ukb_alldx['HF' + i].isna(), np.nan, np.where(
                                pmh_ukb_alldx['HF' + i]==1, (pmh_ukb_alldx["hf_date"] - pmh_ukb_alldx['visit' + i]).dt.days,
                                np.where((pmh_ukb_alldx["lost_fu_date"].notna()) & (pmh_ukb_alldx['lost_fu_date']<pmh_ukb_alldx['visit' + i]),
                                            (pmh_ukb_alldx["lost_fu_date"] - pmh_ukb_alldx['visit' + i]).dt.days,
                                (pd.to_datetime("2021-05-20") - pmh_ukb_alldx['visit' + i]).dt.days)))

    # Different definition for HTN and DM
    pmh_ukb_alldx['htn' + i] = np.where(pmh_ukb_alldx['visit' + i].isna(), np.nan, np.where(
                                    (pmh_ukb_alldx['htn_date'].notna()) & (pmh_ukb_alldx['htn_date']<=pmh_ukb_alldx['visit' + i]), 1, 0))
    pmh_ukb_alldx['dm' + i] = np.where(pmh_ukb_alldx['visit' + i].isna(), np.nan, np.where(
                                    (pmh_ukb_alldx['dm_date'].notna()) & (pmh_ukb_alldx['dm_date']<=pmh_ukb_alldx['visit' + i]), 1, 0))
    pmh_ukb_alldx['ckd' + i] = np.where(pmh_ukb_alldx['visit' + i].isna(), np.nan, np.where(
                                    (pmh_ukb_alldx['ckd_date'].notna()) & (pmh_ukb_alldx['ckd_date']<=pmh_ukb_alldx['visit' + i]), 1, 0))
    
for i in ['visit_2', 'visit_3', 'lost_fu_date']:
    pmh_ukb_alldx[i] = pd.to_datetime(pmh_ukb_alldx[i])
    
for i in ['IHD', 'Time2IHD', 'Stroke', 'Time2Stroke', 'HF', 'Time2HF']:
    pmh_ukb_alldx[i + '_2'] = np.where((pmh_ukb_alldx['lost_fu_date']<pmh_ukb_alldx['visit_2']) | (pmh_ukb_alldx['visit_2'].isna()), 
                                        np.nan, pmh_ukb_alldx[i + '_2'])
    pmh_ukb_alldx[i + '_3'] = np.where((pmh_ukb_alldx['lost_fu_date']<pmh_ukb_alldx['visit_3']) | (pmh_ukb_alldx['visit_3'].isna()), 
                                        np.nan, pmh_ukb_alldx[i + '_3'])

for i in ['IHD_2', 'IHD_3', 'Stroke_2', 'Stroke_3', 'HF_2', 'HF_3', 'htn_2', 'htn_3', 'dm_2', 'dm_3']:
    pmh_ukb_alldx[i] = pmh_ukb_alldx.groupby('eid')[i].transform(max)

for i in ['IHD_2', 'IHD_3', 'Stroke_2', 'Stroke_3', 'HF_2', 'HF_3']:
    pmh_ukb_alldx['Time2' + i] = pmh_ukb_alldx.groupby('eid')['Time2' + i].transform(min)

pmh_ukb_alldx.drop_duplicates(subset='eid', inplace=True)

demo_ukb_data = pmh_ukb_alldx

for i in ['_2', '_3']:
    demo_ukb_data['MACE' + i] = np.where((demo_ukb_data['IHD' + i]==1) | (demo_ukb_data['Stroke' + i]==1) | (demo_ukb_data['HF' + i]==1) | (demo_ukb_data['Death' + i]==1), 1,
                            np.where((demo_ukb_data['IHD' + i]==0) & (demo_ukb_data['Stroke' + i]==0) & (demo_ukb_data['HF' + i]==0) & (demo_ukb_data['Death' + i]==0), 0, np.nan))
    demo_ukb_data['Time2MACE' + i] = np.where(demo_ukb_data['MACE' + i]==1, demo_ukb_data[['Time2IHD' + i, 'Time2Stroke' + i, 'Time2HF' + i, 'Time2Death' + i]].min(axis=1),
                        np.where(demo_ukb_data['MACE' + i]==0, demo_ukb_data[['Time2IHD' + i, 'Time2Stroke' + i,  'Time2HF' + i, 'Time2Death' + i]].max(axis=1), np.nan))    
    
demo_ukb_data.drop(columns=['icd', 'all_icd_date', 'icd_type', 'ihd', 'ihd_date', 'stroke', 'stroke_date', 'heart_failure', 'hf_date'], inplace=True)
demo_ukb_data

# %%
for i in ['primary_HF_2', 'primary_HF_3', 'HF_2', 'HF_3']:
    print(demo_ukb_data[i].value_counts())

# %%
#Lifestyle
lifestyle_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([21001,20116])]['col.name'].tolist()]
bmi_stubname = ukb_key[ukb_key['field.showcase'] == 21001]['col.name'].iloc[0].rsplit('_', 2)[0]
smoke_stubname = ukb_key[ukb_key['field.showcase'] == 20116]['col.name'].iloc[0].rsplit('_', 2)[0]

lifestyle_ukb_data = pd.wide_to_long(df=lifestyle_ukb_data, stubnames=[bmi_stubname,smoke_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
lifestyle_ukb_data[['instance','array']] = lifestyle_ukb_data['instance_array'].str.split("_", expand=True)

lifestyle_ukb_data.rename(columns={'body_mass_index_bmi_f21001':'bmi','smoking_status_f20116':'smoking'},inplace=True)
lifestyle_ukb_data['smoking'].replace({-3:np.nan},inplace=True)
# smoking_coding = dict({0:'Never',1:'Previous',2:'Current'})
lifestyle_ukb_data['smoking'].replace({1: 0, 2: 1}, inplace=True)

for i in ['2', '3']:
    lifestyle_ukb_data['bmi_' + i] = np.where(lifestyle_ukb_data['instance']==i,lifestyle_ukb_data['bmi'],np.nan)
    lifestyle_ukb_data['smoking_' + i] = np.where(lifestyle_ukb_data['instance']==i,lifestyle_ukb_data['smoking'],np.nan)

for i in ['bmi_2', 'bmi_3', 'smoking_2', 'smoking_3']:
    lifestyle_ukb_data[i] = lifestyle_ukb_data.groupby('eid')[i].transform(max)

lifestyle_ukb_data = lifestyle_ukb_data[['eid', 'bmi_2', 'bmi_3', 'smoking_2', 'smoking_3']]

lifestyle_ukb_data.drop_duplicates(subset='eid', inplace=True)
lifestyle_ukb_data

# %%
#Medications

#Medications-Categories
drug_ukb_agg = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([6153, 6177])]['col.name'].tolist()]
drug_stubname_a = ukb_key[ukb_key['field.showcase'] == 6177]['col.name'].iloc[0].rsplit('_', 2)[0] #3Cat medications
drug_stubname_b = ukb_key[ukb_key['field.showcase'] == 6153]['col.name'].iloc[0].rsplit('_', 2)[0] #5Cat medications
drug_ukb_agg = pd.wide_to_long(df=drug_ukb_agg, stubnames=[drug_stubname_a, drug_stubname_b],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
drug_ukb_agg[['instance','array']] = drug_ukb_agg['instance_array'].str.split("_", expand=True)
drug_a_coding = dict({-1:np.nan,-3:np.nan,-7:0,1:'llt',2:'anti_htn',3:'insulin'})
drug_ukb_agg.replace({'medication_for_cholesterol_blood_pressure_or_diabetes_f6177':drug_a_coding},inplace=True)
drug_b_coding = dict({-1:np.nan,-3:np.nan,-7:0,1:'llt',2:'anti_htn',3:'insulin',4:'hrt',5:'ocp'})
drug_ukb_agg.replace({'medication_for_cholesterol_blood_pressure_diabetes_or_take_exogenous_hormones_f6153':drug_b_coding},
                        inplace=True)

def llt(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='llt')|(row[drug_stubname_b]=='llt'):
        return 1
    else:
        return 0
def anti_htn(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='anti_htn')|(row[drug_stubname_b]=='anti_htn'):
        return 1
    else:
        return 0
def insulin(row):
    if pd.isna(row[drug_stubname_a]) & pd.isna(row[drug_stubname_b]):
        return np.nan
    elif (row[drug_stubname_a]=='insulin')|(row[drug_stubname_b]=='insulin'):
        return 1
    else:
        return 0


for i in ['2', '3']:
    drug_ukb_agg['anti_htn_' + i] = drug_ukb_agg.apply(lambda row: anti_htn(row) if row['instance']==i else None, axis=1)
    drug_ukb_agg['llt_' + i] = drug_ukb_agg.apply(lambda row: llt(row) if row['instance']==i else None, axis=1)
    drug_ukb_agg['insulin_' + i] = drug_ukb_agg.apply(lambda row: insulin(row) if row['instance']==i else None, axis=1)

for i in ['anti_htn_2', 'anti_htn_3', 'insulin_2', 'insulin_3', 'llt_2', 'llt_3']:
    drug_ukb_agg[i] = drug_ukb_agg.groupby('eid')[i].transform(max)

drug_ukb_agg.drop_duplicates(subset='eid', inplace=True)
drug_ukb_agg.drop(columns=['instance_array', drug_stubname_a, drug_stubname_b, 'instance', 'array'], inplace=True)
drug_ukb_agg

# %%
#Lab
lab_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([30780,30760,30690,30870,#Lipid
                                                                          30740,30750, #DM
                                                                          30786,30766,30696,30876,
                                                                          30746,30756,
                                                                          30670,30700, #Renal
                                                                          ])]['col.name'].tolist()]

data_coding_4917 = dict({1:'Reportable',2:'low',3:'high',4:'low',5:'high'})
lab_ukb_data.replace({'cholesterol_reportability_f30696_0_0':data_coding_4917,
       'cholesterol_reportability_f30696_1_0':data_coding_4917,
       'glucose_reportability_f30746_0_0':data_coding_4917, 'glucose_reportability_f30746_1_0':data_coding_4917,
       'glycated_haemoglobin_hba1c_reportability_f30756_0_0':data_coding_4917,
       'glycated_haemoglobin_hba1c_reportability_f30756_1_0':data_coding_4917,
       'hdl_cholesterol_reportability_f30766_0_0':data_coding_4917,
       'hdl_cholesterol_reportability_f30766_1_0':data_coding_4917,
       'ldl_direct_reportability_f30786_0_0':data_coding_4917,
       'ldl_direct_reportability_f30786_1_0':data_coding_4917,
       'triglycerides_reportability_f30876_0_0':data_coding_4917,
       'triglycerides_reportability_f30876_1_0':data_coding_4917}, inplace=True)

lab_ukb_data.rename(columns={'cholesterol_f30690_0_0':'cholesterol_0','cholesterol_f30690_1_0':'cholesterol_1',
                             'hdl_cholesterol_f30760_0_0':'hdl_0','hdl_cholesterol_f30760_1_0':'hdl_1',
                             'ldl_direct_f30780_0_0':'ldl_0','ldl_direct_f30780_1_0':'ldl_1',
                             'triglycerides_f30870_0_0':'triglycerides_0','triglycerides_f30870_1_0':'triglycerides_1',
                             'glucose_f30740_0_0':'glucose_0','glucose_f30740_1_0':'glucose_1', 
                             'glycated_haemoglobin_hba1c_f30750_0_0':'hba1c_0',
                             'glycated_haemoglobin_hba1c_f30750_1_0':'hba1c_1',
                             'creatinine_f30700_0_0': 'cr_0',
                             'creatinine_f30700_1_0':'cr_1'},
                             inplace=True)


#Reducing missingness by reportability variables

lab_ukb_data['cholesterol_0'] = np.where((lab_ukb_data['cholesterol_0'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_0_0']=='low'), 
                    0.601, np.where((lab_ukb_data['cholesterol_0'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_0_0']=='high'),
                    15.46, lab_ukb_data['cholesterol_0']))
lab_ukb_data['cholesterol_1'] = np.where((lab_ukb_data['cholesterol_1'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_1_0']=='low'), 
                    0.601, np.where((lab_ukb_data['cholesterol_1'].isna()) & (lab_ukb_data['cholesterol_reportability_f30696_1_0']=='high'),
                    15.46, lab_ukb_data['cholesterol_1']))
lab_ukb_data['glucose_0'] = np.where((lab_ukb_data['glucose_0'].isna()) & (lab_ukb_data['glucose_reportability_f30746_0_0']=='low'), 
                    0.995, np.where((lab_ukb_data['glucose_0'].isna()) & (lab_ukb_data['glucose_reportability_f30746_0_0']=='high'),
                    36.813, lab_ukb_data['glucose_0']))
lab_ukb_data['glucose_1'] = np.where((lab_ukb_data['glucose_1'].isna()) & (lab_ukb_data['glucose_reportability_f30746_1_0']=='low'), 
                    0.995, np.where((lab_ukb_data['glucose_1'].isna()) & (lab_ukb_data['glucose_reportability_f30746_1_0']=='high'),
                    36.813, lab_ukb_data['glucose_1']))
lab_ukb_data['hba1c_0'] = np.where((lab_ukb_data['hba1c_0'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_0_0']=='low'), 
                    15, np.where((lab_ukb_data['hba1c_0'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_0_0']=='high'),
                    515.2, lab_ukb_data['hba1c_0']))
lab_ukb_data['hba1c_1'] = np.where((lab_ukb_data['hba1c_1'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_1_0']=='low'), 
                    15, np.where((lab_ukb_data['hba1c_1'].isna()) & (lab_ukb_data['glycated_haemoglobin_hba1c_reportability_f30756_1_0']=='high'),
                    515.2, lab_ukb_data['hba1c_1']))
lab_ukb_data['hdl_0'] = np.where((lab_ukb_data['hdl_0'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_0_0']=='low'), 
                    0.219, np.where((lab_ukb_data['hdl_0'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_0_0']=='high'),
                    4.401, lab_ukb_data['hdl_0']))
lab_ukb_data['hdl_1'] = np.where((lab_ukb_data['hdl_1'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_1_0']=='low'), 
                    0.219, np.where((lab_ukb_data['hdl_1'].isna()) & (lab_ukb_data['hdl_cholesterol_reportability_f30766_1_0']=='high'),
                    4.401, lab_ukb_data['hdl_1']))
lab_ukb_data['ldl_0'] = np.where((lab_ukb_data['ldl_0'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_0_0']=='low'), 
                    0.219, np.where((lab_ukb_data['ldl_0'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_0_0']=='high'),
                    4.401, lab_ukb_data['ldl_0']))
lab_ukb_data['ldl_1'] = np.where((lab_ukb_data['ldl_1'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_1_0']=='low'), 
                    0.219, np.where((lab_ukb_data['ldl_1'].isna()) & (lab_ukb_data['ldl_direct_reportability_f30786_1_0']=='high'),
                    4.401, lab_ukb_data['ldl_1']))
lab_ukb_data['triglycerides_0'] = np.where((lab_ukb_data['triglycerides_0'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_0_0']=='low'), 
                    0.231, np.where((lab_ukb_data['triglycerides_0'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_0_0']=='high'),
                    11.278, lab_ukb_data['triglycerides_0']))
lab_ukb_data['triglycerides_1'] = np.where((lab_ukb_data['triglycerides_1'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_1_0']=='low'), 
                    0.231, np.where((lab_ukb_data['triglycerides_1'].isna()) & (lab_ukb_data['triglycerides_reportability_f30876_1_0']=='high'),
                    11.278, lab_ukb_data['triglycerides_1']))
                    
lab_ukb_data.drop(columns=['cholesterol_reportability_f30696_0_0', 'cholesterol_reportability_f30696_1_0',
       'glucose_reportability_f30746_0_0', 'glucose_reportability_f30746_1_0',
       'glycated_haemoglobin_hba1c_reportability_f30756_0_0', 'glycated_haemoglobin_hba1c_reportability_f30756_1_0',
       'hdl_cholesterol_reportability_f30766_0_0', 'hdl_cholesterol_reportability_f30766_1_0',
       'ldl_direct_reportability_f30786_0_0', 'ldl_direct_reportability_f30786_1_0',
       'triglycerides_reportability_f30876_0_0', 'triglycerides_reportability_f30876_1_0'], inplace=True)

# Keeping visit_1 and if NA, visit_0
for i in ['cholesterol', 'glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides', 'cr']:
       lab_ukb_data[i] = np.where(lab_ukb_data[i +'_1'].notna(), lab_ukb_data[i +'_1'], lab_ukb_data[i +'_0'])

lab_ukb_data = lab_ukb_data[['eid', 'cholesterol', 'glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides', 'cr']]
lab_ukb_data['glucose'] = lab_ukb_data['glucose']*18.018
lab_ukb_data['cholesterol'] = lab_ukb_data['cholesterol']*38.67
lab_ukb_data['hdl'] = lab_ukb_data['hdl']*38.67
lab_ukb_data['ldl'] = lab_ukb_data['ldl']*38.67
lab_ukb_data['triglycerides'] = lab_ukb_data['triglycerides']*88.57
lab_ukb_data['cr'] = lab_ukb_data['cr']/88.42
lab_ukb_data

# %%
#Vitals
vital_ukb_data = ukb_data[['eid'] + ukb_key[ukb_key['field.showcase'].isin([4079,4080,93,94,95,102
                                                                           ])]['col.name'].tolist()]

sbp_auto_stubname = ukb_key[ukb_key['field.showcase'] == 4080]['col.name'].iloc[0].rsplit('_', 2)[0]
sbp_manual_stubname = ukb_key[ukb_key['field.showcase'] == 93]['col.name'].iloc[0].rsplit('_', 2)[0]
dbp_auto_stubname = ukb_key[ukb_key['field.showcase'] == 4079]['col.name'].iloc[0].rsplit('_', 2)[0]
dbp_manual_stubname = ukb_key[ukb_key['field.showcase'] == 94]['col.name'].iloc[0].rsplit('_', 2)[0]
hr_auto_stubname = ukb_key[ukb_key['field.showcase'] == 102]['col.name'].iloc[0].rsplit('_', 2)[0]
hr_manual_stubname = ukb_key[ukb_key['field.showcase'] == 95]['col.name'].iloc[0].rsplit('_', 2)[0]
vital_ukb_data = pd.wide_to_long(df=vital_ukb_data, stubnames=[sbp_auto_stubname,sbp_manual_stubname,dbp_auto_stubname,
                                                               dbp_manual_stubname,hr_auto_stubname,hr_manual_stubname],
                                                              i='eid', j='instance_array', sep='_', suffix=r'\w+').reset_index()
vital_ukb_data[['instance','array']] = vital_ukb_data['instance_array'].str.split("_", expand=True)

vital_ukb_data['sbp'] = np.where(vital_ukb_data['systolic_blood_pressure_automated_reading_f4080'].notna(),
    vital_ukb_data['systolic_blood_pressure_automated_reading_f4080'], vital_ukb_data['systolic_blood_pressure_manual_reading_f93'])
vital_ukb_data['dbp'] = np.where(vital_ukb_data['diastolic_blood_pressure_automated_reading_f4079'].notna(),
    vital_ukb_data['diastolic_blood_pressure_automated_reading_f4079'], vital_ukb_data['diastolic_blood_pressure_manual_reading_f94'])
vital_ukb_data['hr'] = np.where(vital_ukb_data['pulse_rate_automated_reading_f102'].notna(),
    vital_ukb_data['pulse_rate_automated_reading_f102'], vital_ukb_data['pulse_rate_during_bloodpressure_measurement_f95'])

vital_ukb_data.drop(columns=['instance_array','systolic_blood_pressure_automated_reading_f4080',
                             'systolic_blood_pressure_manual_reading_f93','diastolic_blood_pressure_automated_reading_f4079',
                             'diastolic_blood_pressure_manual_reading_f94','pulse_rate_automated_reading_f102',
                             'pulse_rate_during_bloodpressure_measurement_f95'], inplace=True)

for i in ['2', '3']:
    vital_ukb_data['sbp_' + i] = vital_ukb_data[vital_ukb_data['instance']==i].groupby('eid')['sbp'].transform('mean')
    vital_ukb_data['dbp_' + i] = vital_ukb_data[vital_ukb_data['instance']==i].groupby('eid')['dbp'].transform('mean')
    vital_ukb_data['hr_' + i] = vital_ukb_data[vital_ukb_data['instance']==i].groupby('eid')['hr'].transform('mean')

for i in ['sbp_2', 'sbp_3', 'dbp_2', 'dbp_3', 'hr_2', 'hr_3']:
    vital_ukb_data[i] = vital_ukb_data.groupby('eid')[i].transform('mean')

vital_ukb_data.drop_duplicates(subset='eid', inplace=True)
vital_ukb_data.drop(columns=['instance','array','sbp','dbp','hr'], inplace=True)

vital_ukb_data

# %%
wf = ukb_ecg.merge(demo_ukb_data, on='eid', how='left').merge(lifestyle_ukb_data, on='eid', how='left').merge(
    drug_ukb_agg, on='eid', how='left').merge(lab_ukb_data, on='eid', how='left').merge(vital_ukb_data, on='eid', how='left')
for i in ['visit', 'age', 'Death', 'Time2Death', 'bmi', 'smoking', 'anti_htn', 'llt', 'insulin', 'htn', 'dm', 'ckd', 'sbp', 'dbp', 'hr',
          
          'IHD', 'Time2IHD', 'Stroke', 'Time2Stroke', 'HF', 'Time2HF', 'MACE', 'Time2MACE',
          
          'primary_IHD', 'primary_Time2IHD', 'primary_Stroke', 'primary_Time2Stroke', 'primary_HF', 'primary_Time2HF', 'primary_MACE', 'primary_Time2MACE']:
    wf[i] = np.where(wf['ecg_instance']==2, wf[i + '_2'], wf[i + '_3'])

wf['primary_secondary_hf_mortality'] = wf[wf['Death']==1]['primary_secondary_hf_mortality'].fillna(0)

wf['HF_Death_All_HF_Hosp'] = np.where((wf['primary_secondary_hf_mortality']==1) | (wf['HF']==1), 1,
                            np.where(((wf['Death']==0) | (wf['primary_secondary_hf_mortality']==0)) & (wf['HF']==0), 0, np.nan))
wf['Time2HF_Death_All_HF_Hosp'] = np.where(wf['HF_Death_All_HF_Hosp']==1, wf[['Time2HF', 'Time2Death']].min(axis=1),
                        np.where(wf['HF_Death_All_HF_Hosp']==0, wf[['Time2HF', 'Time2Death']].max(axis=1), np.nan))

wf['HF_Death_Primary_HF_Hosp'] = np.where((wf['primary_secondary_hf_mortality']==1) | (wf['primary_HF']==1), 1,
                            np.where(((wf['Death']==0) | (wf['primary_secondary_hf_mortality']==0)) & (wf['primary_HF']==0), 0, np.nan))
wf['Time2HF_Death_Primary_HF_Hosp'] = np.where(wf['HF_Death_Primary_HF_Hosp']==1, wf[['primary_Time2HF', 'Time2Death']].min(axis=1),
                        np.where(wf['HF_Death_Primary_HF_Hosp']==0, wf[['primary_Time2HF', 'Time2Death']].max(axis=1), np.nan))

for i in ['Time2IHD', 'Time2Stroke', 'Time2HF', 'Time2MACE', 'Time2HF_Death_All_HF_Hosp',
          
          'primary_Time2IHD', 'primary_Time2Stroke', 'primary_Time2HF', 'primary_Time2MACE', 'Time2HF_Death_Primary_HF_Hosp']:
    wf[i] = np.where(wf[i]>wf['Time2Death'], wf['Time2Death'], wf[i])

# Defining HTN and DM
wf['htn_pcp_hf'] = np.where((wf['htn']==1) | (wf['anti_htn']==1), 1, 0)
wf['dm_pcp_hf'] = np.where((wf['dm']==1) | (wf['insulin']==1), 1, 0)

# Removing withdrawals
withdrawal = pd.read_csv('/path_to_ukb/ukb_withdrawals.csv')
wf = wf[~wf['eid'].isin(withdrawal['eid'].to_list())]
wf.drop(columns='Unnamed: 0', inplace=True)

wf.drop(columns=wf.columns[wf.columns.str.contains('_2|_3')].to_list(), inplace=True)
wf

# %%
# calculate GFR based on CKD-EPI formula
def calculate_egfr(age, scr, sex):
    """
    Calculate the estimated glomerular filtration rate (eGFR).

    Parameters:
    age (float): Age of the patient in years.
    scr (float): Serum creatinine level (mg/dL).
    sex (int): Sex of the patient (1: 'male' or 0: 'female').

    Returns:
    float: The estimated glomerular filtration rate (eGFR).
    """
    # Constants based on sex
    if sex == 0:
        k = 0.7
        alpha = -0.241
        female_multiplier = 1.012
    elif sex == 1:
        k = 0.9
        alpha = -0.302
        female_multiplier = 1.0
    else:
        raise ValueError("Sex must be 1: 'male' or 0: 'female'")

    # Calculate min and max values
    min_value = min(scr / k, 1)
    max_value = max(scr / k, 1)

    # Calculate eGFR
    egfr = 142 * (min_value ** alpha) * (max_value ** -1.200) * (0.9938 ** age) * female_multiplier

    return egfr

# Example usage
age = 55
scr = 1.1
sex = 'female'

wf['egfr_ckd_epi'] = wf.apply(lambda row: calculate_egfr(
    age=row['age'], 
    scr=row['cr'], 
    sex=row['sex']), axis=1)

# %%
wf[['age', 'cr', 'sex', 'gender', 'egfr_ckd_epi']].head()

# %%
# Apply the function to the DataFrame
wf['sex_group'] = np.where(wf['sex']==0, 'Women', 'Men')
wf['prevent_hf_10yr_risk'] = wf.apply(lambda row: calculate_prevent_hf_10yr_risk(
    age=row['age'], 
    sbp=row['sbp'], 
    diabetes=row['dm'], 
    gfr=row['egfr_ckd_epi'], 
    bmi=row['bmi'], 
    current_smoker=row['smoking'], 
    sex_group=row['sex_group'], 
    antihypertensive_med=row['anti_htn'], 
    ), axis=1)

# %%
wf['pcp_hf_risk'] = wf.apply(lambda row: calculate_hf_risk(
    age=row['age'], 
    systolic_bp=row['sbp'], 
    glucose=row['glucose'], 
    total_cholesterol=row['cholesterol'], 
    hdl_c=row['hdl'], 
    bmi=row['bmi'], 
    qrs_duration=row['QRS'], 
    smoker=row['smoking'], 
    race_sex_group=row['race_sex_group'], 
    treated_bp=row['htn_pcp_hf'], 
    treated_glucose=row['dm_pcp_hf']), axis=1)


af_variations = ['afib', 'aflutter', 'flutter', 'atrial fibrillation',
                 'atrialfibrillation', 'atrial flutter', 'atrialflutter',
                 'af ', 'af/']
wf['AF_ecg_dx'] = wf['diagnoses'].str.contains('|'.join(af_variations), case=False)
wf['LBBB_ecg_dx'] = wf['diagnoses'].str.contains('left bundle branch block', case=False)

wf.to_csv('/path_to_ukb/ukb_df_processed.csv', index=False)
#%%