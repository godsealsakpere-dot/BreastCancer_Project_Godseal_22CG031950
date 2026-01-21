import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1. Load the Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target 

# Note on Target Encoding in sklearn dataset:
# 0 = Malignant (Cancerous), 1 = Benign (Safe)
# We will verify this using data.target_names

# 2. Feature Selection
# Instructions: Select exactly 5 features.
# Selected: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = df[selected_features]
y = df['diagnosis']

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing & Model Pipeline
# Scaling is MANDATORY for SVM. We use a Pipeline to bundle the Scaler and the Model.
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature Scaling
    ('classifier', SVC(kernel='linear', random_state=42)) # Algorithm: SVM
])

# 5. Train the Model
print("Training Model...")
pipeline.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = pipeline.predict(X_test)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Save the Model
# Ensure the folder exists
if not os.path.exists('model'):
    os.makedirs('model')

model_filename = 'model/breast_cancer_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")

# 8. Reload Demonstration
loaded_model = joblib.load(model_filename)
result = loaded_model.score(X_test, y_test)
print(f"Reloaded model accuracy check: {result:.4f}")