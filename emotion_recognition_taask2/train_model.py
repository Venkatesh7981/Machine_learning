import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load features
df = pd.read_csv('model/features.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/emotion_model.pkl')

print("Model trained and saved.")
