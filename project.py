from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load datasets
tap_working_memory = pd.read_csv("TAP-Working Memory.csv")
meta_data = pd.read_csv("META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")

# Merge datasets on ID
merged_data = tap_working_memory.merge(meta_data[['ID', 'Age']], on='ID', how='inner')

# Convert relevant columns to numeric
columns_to_convert = ['TAP_WM_3', 'TAP_WM_5', 'TAP_WM_8', 'TAP_WM_10', 'TAP_WM_12']
for col in columns_to_convert:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Drop unnecessary or completely missing columns
merged_data = merged_data.drop(columns=['TAP_WM_12'])

# Drop duplicates and missing values
merged_data = merged_data.drop_duplicates().dropna()

def classify_age_group(age):
    younger_ages = ("20-25", "25-30")
    older_ages = ("30-35", "35-40", "55-60", "60-65", "65-70", "70-75", "75-80")

    if age in younger_ages:
        return "young"
    elif age in older_ages:
        return "old"
    else:
        return "unknown"

