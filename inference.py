import joblib
from utils import extract_features

# Load trained model
model_path = "../model/resume_crf_model.pkl"
crf = joblib.load(model_path)

def parse_resume(tokens):
    features = [extract_features([token])[0] for token in tokens]
    prediction = crf.predict([features])[0]
    return list(zip(tokens, prediction))

# Example usage
if __name__ == "__main__":
    test_tokens = ["John", "Doe", "email", "johndoe@gmail.com", "Phone", "1234567890"]
    parsed = parse_resume(test_tokens)
    for token, label in parsed:
        print(f"{token} --> {label}")
