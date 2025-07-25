from sklearn_crfsuite import metrics
import joblib
from utils import load_data, extract_features, extract_labels

# Paths
test_data_path = "../data/test.txt"
model_path = "../model/resume_crf_model.pkl"

# Load data
sentences = load_data(test_data_path)

# Extract features and labels
X_test = [extract_features(s) for s in sentences]
y_test = [extract_labels(s) for s in sentences]

# Load model
crf = joblib.load(model_path)

# Predict
y_pred = crf.predict(X_test)

# Evaluate
print(metrics.flat_classification_report(y_test, y_pred, digits=3))
