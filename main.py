# Iris Flower Classification - main.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("C:\\Users\\MEGHANA\\Downloads\\archive\\IRIS.csv")  
df.columns = df.columns.str.strip()

# Print column names for debugging
print("✅ Columns in dataset:", list(df.columns))

# Encode the correct column: 'species' (all lowercase)
if 'species' not in df.columns:
    raise ValueError("❌ 'species' column not found in dataset. Please check the column names!")

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split features and target
X = df.drop('species', axis=1)
y = df['species']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Visualization
sns.pairplot(df.assign(species=le.inverse_transform(df['species'])), hue='species')
plt.suptitle("Iris Flower Classification - Pairplot", y=1.02)
plt.show()
