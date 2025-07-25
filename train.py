import os
import numpy as np
from sklearn_crfsuite import CRF
import joblib
from utils import load_data, extract_features, extract_labels

# Paths
train_data_path = "../data/train.txt"
model_path = "../model/resume_crf_model.pkl"

# Load data
sentences = load_data(train_data_path)

# Extract features and labels
X_train = [extract_features(s) for s in sentences]
y_train = [extract_labels(s) for s in sentences]

# Train CRF
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
crf.fit(X_train, y_train)

# Save model
os.makedirs("../model", exist_ok=True)
joblib.dump(crf, model_path)

print(f"Model trained and saved to {model_path}")
