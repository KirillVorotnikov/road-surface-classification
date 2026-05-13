import joblib
import pandas as pd

model, scaler = joblib.load('models\svm_model.joblib')

X_new_scaled = scaler.transform(X_new)

predictions = model.predict(X_new_scaled)

print(predictions)