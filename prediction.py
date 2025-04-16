
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Load test data
X_test = pd.read_csv('X_test.csv')

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Predict
predictions = model.predict(X_test_scaled)

# Print and save predictions
print("Predictions:", predictions[:10])
pd.DataFrame(predictions, columns=["prediction"]).to_csv("predictions.csv", index=False)
