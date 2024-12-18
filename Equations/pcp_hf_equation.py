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
coefficients = {
    'White Male': {
        'Ln Age': 41.94,
        'Ln Age Squared': -0.88,
        'Ln Treated Systolic BP': 1.03,
        'Ln Age x Ln Treated Systolic BP': 0, #np.nan should be 0
        'Ln Untreated Systolic BP': 0.91,
        'Ln Age x Ln Untreated Systolic BP': 0, #np.nan should be 0
        'Current Smoker': 0.74,
        'Ln Age x Current Smoker': 0, #np.nan should be 0
        'Ln Treated glucose': 0.90,
        'Ln Untreated glucose': 0.78,
        'Ln Total Cholesterol': 0.49,
        'Ln HDL-C': -0.44,
        'Ln BMI': 37.2,
        'Ln Age x Ln BMI': -8.83,
        'Ln QRS duration': 0.63,
        'Mean Coefficient Value': 171.5,
        'Baseline Survival': 0.98752
    },
    'White Female': {
        'Ln Age': 20.55,
        'Ln Age Squared': 0, #np.nan should be 0
        'Ln Treated Systolic BP': 12.95,
        'Ln Age x Ln Treated Systolic BP': -2.96,
        'Ln Untreated Systolic BP': 11.86,
        'Ln Age x Ln Untreated Systolic BP': -2.73,
        'Current Smoker': 11.02,
        'Ln Age x Current Smoker': -2.50,
        'Ln Treated glucose': 1.04,
        'Ln Untreated glucose': 0.91,
        'Ln Total Cholesterol': 0, #np.nan should be 0
        'Ln HDL-C': -0.07,
        'Ln BMI': 1.33,
        'Ln Age x Ln BMI': 0, #np.nan should be 0
        'Ln QRS duration': 1.06,
        'Mean Coefficient Value': 99.73,
        'Baseline Survival': 0.99348
    },
    'Black Male': {
        'Ln Age': 2.88,
        'Ln Age Squared': 0, #np.nan should be 0
        'Ln Treated Systolic BP': 2.31,
        'Ln Age x Ln Treated Systolic BP': 0, #np.nan should be 0
        'Ln Untreated Systolic BP': 2.17,
        'Ln Age x Ln Untreated Systolic BP': 0, #np.nan should be 0
        'Current Smoker': 1.66,
        'Ln Age x Current Smoker': -0.25,
        'Ln Treated glucose': 0.64,
        'Ln Untreated glucose': 0.58,
        'Ln Total Cholesterol': 0, #np.nan should be 0
        'Ln HDL-C': -0.81,
        'Ln BMI': 1.16,
        'Ln Age x Ln BMI': 0, #np.nan should be 0
        'Ln QRS duration': 0.73,
        'Mean Coefficient Value': 28.73,
        'Baseline Survival': 0.98295
    },
    'Black Female': {
        'Ln Age': 51.75,
        'Ln Age Squared': 0, #np.nan should be 0
        'Ln Treated Systolic BP': 29.0,
        'Ln Age x Ln Treated Systolic BP': -6.59,
        'Ln Untreated Systolic BP': 28.18,
        'Ln Age x Ln Untreated Systolic BP': -6.42,
        'Current Smoker': 0.76,
        'Ln Age x Current Smoker': 0, #np.nan should be 0
        'Ln Treated glucose': 0.97,
        'Ln Untreated glucose': 0.80,
        'Ln Total Cholesterol': 0.32,
        'Ln HDL-C': 0, #np.nan should be 0
        'Ln BMI': 21.24,
        'Ln Age x Ln BMI': -5.0,
        'Ln QRS duration': 1.27,
        'Mean Coefficient Value': 233.9,
        'Baseline Survival': 0.99260
    },
    'Others': {
        'Ln Age': 0, #np.nan should be 0
        'Ln Age Squared': 0, #np.nan should be 0
        'Ln Treated Systolic BP': 0, #np.nan should be 0
        'Ln Age x Ln Treated Systolic BP': 0, #np.nan should be 0
        'Ln Untreated Systolic BP': 0, #np.nan should be 0
        'Ln Age x Ln Untreated Systolic BP': 0, #np.nan should be 0
        'Current Smoker': 0, #np.nan should be 0
        'Ln Age x Current Smoker': 0, #np.nan should be 0
        'Ln Treated glucose': 0, #np.nan should be 0
        'Ln Untreated glucose': 0, #np.nan should be 0
        'Ln Total Cholesterol': 0, #np.nan should be 0
        'Ln HDL-C': 0, #np.nan should be 0
        'Ln BMI': 0, #np.nan should be 0
        'Ln Age x Ln BMI': 0, #np.nan should be 0
        'Ln QRS duration': 0, #np.nan should be 0
        'Mean Coefficient Value': 0, #np.nan should be 0
        'Baseline Survival': 0, #np.nan should be 0
    }
}



def calculate_hf_risk(age, systolic_bp, glucose, total_cholesterol, hdl_c, bmi, qrs_duration, smoker, race_sex_group, treated_bp=False, treated_glucose=False):
    """
    Calculate the 10-year risk of heart failure based on the given parameters.

    Parameters:
    age (float): The age of the individual.
    systolic_bp (float): The systolic blood pressure.
    glucose (float): The glucose level.
    total_cholesterol (float): The total cholesterol level.
    hdl_c (float): The HDL cholesterol level.
    bmi (float): The body mass index.
    qrs_duration (float): The QRS duration from ECG.
    smoker (bool): Smoking status (True if current smoker, False otherwise).
    race_sex_group (str): The race and sex group as defined in the coefficients dictionary.
    treated_bp (bool): If the blood pressure is treated.
    treated_glucose (bool): If the glucose is treated.

    Returns:
    float: The estimated 10-year risk of heart failure.
    """

    if race_sex_group == 'Others':
        return np.nan
    
    # Check if any input is NaN (np.nan) and return np.nan for the function
    inputs = [age, systolic_bp, glucose, total_cholesterol, hdl_c, bmi, qrs_duration, smoker, treated_bp, treated_glucose]
    
    if any(pd.isna(value) for value in inputs):
        return np.nan

    # Access the specific coefficients for the race and sex group
    coeffs = coefficients[race_sex_group]

    # Calculate the individual components of IndX
    ind_x = (
        # Ln Age
            coeffs['Ln Age'] * np.log(age) +
        # Ln Age Squared
            coeffs['Ln Age Squared'] * np.log(age)**2 +
        # Ln Treated Systolic BP
            coeffs['Ln Treated Systolic BP'] * np.log(systolic_bp) * treated_bp +
        # Ln Age x Ln Treated Systolic BP
            coeffs['Ln Age x Ln Treated Systolic BP'] * np.log(age) * np.log(systolic_bp) * treated_bp +
        # Ln Untreated Systolic BP
            coeffs['Ln Untreated Systolic BP'] * np.log(systolic_bp) * (not treated_bp) +
        # Ln Age x Ln Untreated Systolic BP
            coeffs['Ln Age x Ln Untreated Systolic BP'] *  np.log(age) * np.log(systolic_bp) * (not treated_bp) +
        # Current Smoker
            coeffs['Current Smoker'] * smoker +
        # Ln Age x Current Smoker
            coeffs['Ln Age x Current Smoker'] * np.log(age) * smoker +
        # Ln Treated glucose
            coeffs['Ln Treated glucose'] * np.log(glucose) * treated_glucose +
        # Ln Untreated glucose
            coeffs['Ln Untreated glucose'] * np.log(glucose) * (not treated_glucose) +
        # Ln Total Cholesterol
            coeffs['Ln Total Cholesterol'] * np.log(total_cholesterol) +
        # Ln HDL-C
            coeffs['Ln HDL-C'] * np.log(hdl_c) +
        # Ln BMI
            coeffs['Ln BMI'] * np.log(bmi) +
        # Ln Age x Ln BMI
            coeffs['Ln Age x Ln BMI'] * np.log(age) * np.log(bmi) + 
        # Ln QRS duration
            coeffs['Ln QRS duration'] * np.log(qrs_duration) 

    )

    # Calculate the risk
    mean_cv = coeffs['Mean Coefficient Value']
    baseline_survival = coeffs['Baseline Survival']
    pcp_hf_risk = 1 - baseline_survival**(np.exp(ind_x - mean_cv))

    return pcp_hf_risk


#%%
'''
# Example Usage
age=50
systolic_bp=130
glucose=100 
total_cholesterol=200
hdl_c=50 
bmi=28 
qrs_duration=120 
smoker=False
race_sex_group='Black Female'
treated_bp=False 
treated_glucose=False
coeffs = coefficients[race_sex_group]


risk = calculate_hf_risk(age=50, systolic_bp=130, glucose=100, total_cholesterol=200,
                         hdl_c=50, bmi=28, qrs_duration=120, smoker=False,
                         race_sex_group='Black Female', treated_bp=False, treated_glucose=False)
print(f"The estimated 10-year risk of heart failure is: {pcp_hf_risk:.6%}")
'''

# %%
