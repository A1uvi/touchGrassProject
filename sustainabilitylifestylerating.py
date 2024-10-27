import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import json
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib  # For saving the model


# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data/lifestyle_sustainability_data.csv')

# Drop columns difficult to use
df = df.drop(['ParticipantID', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption','HomeSize'], axis=1)

# Drop null rows
df = df.dropna()

print(df.info())
print(df.describe().T)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isna().sum())

# Encoding categorical features
encoding_mappings = {}
for column in df.columns:
    if df[column].dtype == 'object' or df[column].dtype == 'bool':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoding_mappings[column] = {str(key): int(value) for key, value in zip(le.classes_, le.transform(le.classes_))}

# Save the DataFrame with encoded columns to a new CSV file
df.to_csv('static/assets/encoded_file.csv', index=False)

# Save the encoding mappings to a JSON file
with open('static/assets/encoding_mappings.json', 'w') as f:
    json.dump(encoding_mappings, f)

print("Encoding complete and mappings saved.")

# Compute the correlation matrix and plot the heatmap
numeric_data = df.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of All Features')
plt.savefig('static/assets/img/sustainabilitylifestyleratingheatmap.png')

# Separate features and target
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column

print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the Random Forest model
joblib.dump(rf_model, 'models/sustainnability_model.pkl')

# Plotting (you can adjust the aesthetics here)
plt.figure(figsize=(10, 6))
plt.barh(range(len(rf_model.feature_importances_)), rf_model.feature_importances_, align='center')
plt.yticks(range(len(rf_model.feature_importances_)), X.columns)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('feature_importances.png')  # Save the plot as an image
plt.show()

print(X_train.columns)