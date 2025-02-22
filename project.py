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



# Apply function to create Age_Group column
merged_data['Age_Group'] = merged_data['Age'].apply(classify_age_group)


# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Define features (X) and target labels (y)
X = merged_data.drop(columns=['ID', 'Age', 'Age_Group'])
y = merged_data['Age_Group']
# Ensure there are no NaN values in y
print(y.isna().sum())
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# Train and evaluate models
best_model_name = None
best_accuracy = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')

print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.2f}")

# Confusion matrix and classification report for the best model 
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


