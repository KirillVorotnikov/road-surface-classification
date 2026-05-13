import os
import joblib

def load_model(model_name="svm_model_1sec.joblib"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
    model, scaler = joblib.load(model_path)
    return model, scaler