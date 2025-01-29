import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Loading the TAP_Working_Memory dataset
tap_working_memory = pd.read_csv('TAP-Working Memory.csv')
# Cleaning the data
columns_to_convert = ['TAP_WM_3', 'TAP_WM_5', 'TAP_WM_8', 'TAP_WM_10', 'TAP_WM_12']


for col in columns_to_convert:
    tap_working_memory[col] = pd.to_numeric(tap_working_memory[col], errors='coerce')

    # Drop unnecessary or completely missing columns
tap_working_memory = tap_working_memory.drop(columns=['TAP_WM_12'])
# Drop rows with missing values
cleaned_data = tap_working_memory.dropna()

# Assigning age groups based on index range
cleaned_data['Age_Group'] = 'young'
cleaned_data.loc[153:, 'Age_Group'] = 'older'

# Define features (X) and target labels (y)
X = cleaned_data.drop(columns=['ID', 'Age_Group'])
y = cleaned_data['Age_Group']

# Encode labels to binary (0 = young, 1 = older)
y = y.map({'young': 0, 'older': 1})

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}
