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
# Function to calculate the PREVENT risk
def calculate_prevent_hf_10yr_risk(age, sbp, diabetes, current_smoker, bmi, gfr, sex_group, antihypertensive_med=False):
    """
    Calculate the 10-year risk of heart failure using the PREVENT tool.

    Parameters:
    age (float): The age of the individual.
    sbp (float): Systolic blood pressure.
    diabetes (bool): If the individual has diabetes (True/False).
    current_smoker (bool): Smoking status (True if current smoker, False otherwise).
    bmi (float): Body mass index.
    gfr (float): Glomerular filtration rate.
    sex_group (str): 'Men' or 'Women' to determine which coefficients to use.
    antihypertensive_med (bool): If the individual uses antihypertensive medication (True/False).

    Returns:
    float: The estimated 10-year heart failure risk.
    """

    # Check for missing sex_group and handle appropriately
    if pd.isnull(sex_group):
        return np.nan

    # Ensure that sex_group is a string and capitalize it
    sex_group = str(sex_group).capitalize()
    if sex_group not in ['Men', 'Women']:
        return np.nan  # Return np.nan for invalid sex_group

    # List of required numerical inputs
    numerical_inputs = [age, sbp, bmi, gfr]
    # Check for missing numerical inputs
    if any(pd.isnull(val) for val in numerical_inputs):
        return np.nan  # Return np.nan if any numerical input is missing

    # Ensure that boolean variables are properly handled
    # If missing, consider as False or handle as per your requirement
    diabetes = False if pd.isnull(diabetes) else diabetes
    current_smoker = False if pd.isnull(current_smoker) else current_smoker
    antihypertensive_med = False if pd.isnull(antihypertensive_med) else antihypertensive_med

    # Convert boolean inputs to integers (0 or 1)
    try:
        diabetes = int(bool(diabetes))
        current_smoker = int(bool(current_smoker))
        antihypertensive_med = int(bool(antihypertensive_med))
    except (ValueError, TypeError):
        return np.nan  # Return np.nan if conversion fails

    if sex_group == 'Women':
        # Coefficients for Women
        intercept = -4.310409
        age_coef = 0.8998235
        min_sbp_coef = -0.4559771
        max_sbp_coef = 0.3576505
        diabetes_coef = 1.038346
        current_smoker_coef = 0.583916
        min_bmi_coef = -0.0072294
        max_bmi_coef = 0.2997706
        min_gfr_coef = 0.7451638
        max_gfr_coef = 0.0557087
        antihypertensive_med_coef = 0.3534442
        # Interaction coefficients
        interaction1_coef = -0.0981511  # Antihypertensive med * (max SBP - 130)/20
        interaction2_coef = -0.0946663  # (Age - 55)/10 * (max SBP - 130)/20
        interaction3_coef = -0.3581041  # (Age - 55)/10 * Diabetes
        interaction4_coef = -0.1159453  # (Age - 55)/10 * current_smoker
        interaction5_coef = -0.003878   # (Age - 55)/10 * (max BMI - 30)/5
        interaction6_coef = -0.1884289  # (Age - 55)/10 * (min eGFR - 60)/(-15)
    else:
        # Coefficients for Men
        intercept = -3.946391
        age_coef = 0.8972642
        min_sbp_coef = -0.6811466
        max_sbp_coef = 0.3634461
        diabetes_coef = 0.923776
        current_smoker_coef = 0.5023736
        min_bmi_coef = -0.0485841
        max_bmi_coef = 0.3726929
        min_gfr_coef = 0.6926917
        max_gfr_coef = 0.0251827
        antihypertensive_med_coef = 0.2980922
        # Interaction coefficients
        interaction1_coef = -0.0497731  # Antihypertensive med * (max SBP - 130)/20
        interaction2_coef = -0.1289201  # (Age - 55)/10 * (max SBP - 130)/20
        interaction3_coef = -0.3040924  # (Age - 55)/10 * Diabetes
        interaction4_coef = -0.1401688  # (Age - 55)/10 * current_smoker
        interaction5_coef = 0.0068126   # (Age - 55)/10 * (max BMI - 30)/5
        interaction6_coef = -0.1797778  # (Age - 55)/10 * (min eGFR - 60)/(-15)

    # Calculations
    age_term = age_coef * (age - 55) / 10

    min_sbp = min(sbp, 110)
    min_sbp_term = min_sbp_coef * (min_sbp - 110) / 20

    max_sbp = max(sbp, 110)
    max_sbp_term = max_sbp_coef * (max_sbp - 130) / 20

    diabetes_term = diabetes_coef * int(diabetes)
    current_smoker_term = current_smoker_coef * int(current_smoker)

    min_bmi = min(bmi, 30)
    min_bmi_term = min_bmi_coef * (min_bmi - 25) / 5

    max_bmi = max(bmi, 30)
    max_bmi_term = max_bmi_coef * (max_bmi - 30) / 5

    min_gfr = min(gfr, 60)
    min_gfr_term = min_gfr_coef * (min_gfr - 60) / (-15)

    max_gfr = max(gfr, 60)
    max_gfr_term = max_gfr_coef * (max_gfr - 90) / (-15)

    antihypertensive_med_term = antihypertensive_med_coef * int(antihypertensive_med)

    # Interaction terms
    interaction1_term = interaction1_coef * int(antihypertensive_med) * (max_sbp - 130) / 20
    interaction2_term = interaction2_coef * (age - 55) / 10 * (max_sbp - 130) / 20
    interaction3_term = interaction3_coef * (age - 55) / 10 * int(diabetes)
    interaction4_term = interaction4_coef * (age - 55) / 10 * int(current_smoker)
    interaction5_term = interaction5_coef * (age - 55) / 10 * (max_bmi - 30) / 5
    interaction6_term = interaction6_coef * (age - 55) / 10 * (min_gfr - 60) / (-15)

    # Sum all terms to calculate log-odds
    log_odds = (
        intercept +
        age_term +
        min_sbp_term +
        max_sbp_term +
        diabetes_term +
        current_smoker_term +
        min_bmi_term +
        max_bmi_term +
        min_gfr_term +
        max_gfr_term +
        antihypertensive_med_term +
        interaction1_term +
        interaction2_term +
        interaction3_term +
        interaction4_term +
        interaction5_term +
        interaction6_term
    )

    # Calculate the risk
    prevent_risk = np.exp(log_odds) / (1 + np.exp(log_odds))

    return prevent_risk

#%%
# Example usage:
age = 50
sbp = 145
diabetes = True  # True or False
current_smoker = False   # True or False
bmi = 28
gfr = 95
sex_group = 'Men'
antihypertensive_med = False  # True or False

risk = calculate_prevent_hf_10yr_risk(age, sbp, diabetes, current_smoker, bmi, gfr, sex_group, antihypertensive_med)
print(f"The estimated 10-year risk of heart failure is: {risk:.6%}")

# %%
