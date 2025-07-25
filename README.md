# resume-parser-bilstm-crf

```markdown
# Resume Parsing using BiLSTM and CRFs

This repository presents an end-to-end system for parsing resumes and extracting structured information such as name, contact details, education, and skills. The project leverages BiLSTM (Bidirectional Long Short-Term Memory) and CRFs (Conditional Random Fields) to perform Named Entity Recognition (NER) across heterogeneous resume formats.

## Requirements

- Python 3.8+
- CRF++ or sklearn-crfsuite
- Libraries: pandas, sklearn, spacy, nltk
- Pretrained word embeddings (GloVe recommended)
- Jupyter Notebook (for training and testing interface)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/resume-parser-bilstm-crf.git
cd resume-parser-bilstm-crf
```

Install the dependencies:

```
pip install -r requirements.txt
```

Make sure CRF++ or sklearn-crfsuite is installed properly:
```
# If using CRF++
sudo apt install crf++

```

## Dataset:

Place your annotated resume dataset under the data/ folder. The dataset should follow CoNLL-style token-label format like:
```
John       B-NAME
Doe        I-NAME
is         O
a          O
Python     B-SKILL
developer  I-SKILL
```

## Model Architecture:

- Embedding Layer (GloVe)
- BiLSTM Layer
- CRF Layer

Each token is represented by a word embedding and passed through a BiLSTM. The CRF layer then models the dependencies between labels for more accurate sequence predictions.


## Training:

Run the training script:
```
python train.py --train data/train.conll --test data/test.conll --epochs 10
```
You can configure batch size, learning rate, and embedding dimensions via arguments in the script.

## Evaluation:

After training, you’ll get precision, recall, and F1-score metrics:
```
python evaluate.py --model saved_model.pkl --test data/test.conll
```
This will output performance for each entity type (e.g., NAME, SKILL, EDUCATION, etc.).

## Example:
```
from inference import parse_resume
output = parse_resume("sample_resume.txt")
print(output)
```

## Output (JSON):
```
{
  "Name": "John Doe",
  "Email": "john.doe@example.com",
  "Education": ["B.Tech in Computer Science"],
  "Skills": ["Python", "TensorFlow", "Django"]
}
```

## Folder Structure:
```
resume-parser-bilstm-crf/
│
├── data/                     # Training and test datasets
├── embeddings/               # GloVe or other embeddings
├── model/                    # Saved model weights
├── scripts/
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation logic
│   └── inference.py          # Resume parsing function
├── notebooks/                # Jupyter notebooks for EDA & testing
├── requirements.txt
└── README.md
```

## Tech Stack:
- Python
- BiLSTM + CRF
- scikit-learn
- spaCy / nltk
- CRF++ / sklearn-crfsuite

## Use Cases:
- HR Automation Systems
- Resume Screening Tools
- Job Recommendation Engines
- Skill Tagging Platforms

