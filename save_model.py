import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset from the Excel file
file_path = r'C:\Users\Sai Dhanush (ND)\OneDrive - Ekasila Educational Society\Desktop\sr4\dataset.csv.xlsx'  # Update the path if needed
df = pd.read_excel(file_path)

# Check the first few rows to understand the dataset structure
print(df.head())

# Assuming the dataset has columns such as 'Mother's Age', 'Maternal Health Condition', 'Child Birth Weight', etc.
# You might need to adjust the feature names based on your actual dataset

# Example: Convert categorical columns (if any) to numeric values
df['Maternal Health Condition'] = df['Maternal Health Condition'].map({'Healthy': 0, 'High Blood Pressure': 1, 'Diabetes': 2, 'Anemia': 3})

# Define features and target
# Replace these feature names with the actual column names in your dataset
X = df[['Mother\'s Age', 'Maternal Health Condition', 'Child Birth Weight', 'Number of Previous Children']]  # Update as needed
y = df['Mortality (Target)']  # Corrected target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler to .pkl files
pickle.dump(model, open('C:/Users/Sai Dhanush (ND)/OneDrive - Ekasila Educational Society/Desktop/sr4/model.pkl', 'wb'))
pickle.dump(scaler, open('C:/Users/Sai Dhanush (ND)/OneDrive - Ekasila Educational Society/Desktop/sr4/scaler.pkl', 'wb'))

print("Model and scaler saved successfully!")
