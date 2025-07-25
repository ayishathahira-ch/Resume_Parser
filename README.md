# 📄 Resume Parser using BiLSTM + CRF

This project uses Conditional Random Fields (CRF) to extract structured information from resumes, such as:

- Name
- Contact Details
- Education
- Skills
- Experience

## 🔧 Features

- Uses CRF model with engineered features
- Supports raw resume input in text format
- Outputs structured JSON-like key-value pairs

## 🧰 Tech Stack

- Python
- sklearn-crfsuite
- spaCy
- CRF++
- Jupyter Notebooks

## 📂 Structure

- `src/` - All main source code
- `data/` - Training data in CoNLL format
- `notebooks/` - Exploration & EDA
- `models/` - Saved CRF model

## 🚀 How to Run

1. Install dependencies

```bash
pip install -r requirements.txt

