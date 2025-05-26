# 📰 Fake News Detection with Python and Machine Learning

This project demonstrates how to build a machine learning pipeline to classify news articles as **REAL** or **FAKE** using Python, pandas, and scikit-learn. The workflow includes data loading, preprocessing, and splitting for model training and evaluation.

---

## 📁 Dataset

The dataset (`news.csv`) contains 7,796 news articles with the following columns:

| Column Name  | Description                      |
| ------------ | -------------------------------- |
| `Unnamed: 0` | Article ID or index              |
| `title`      | Headline of the article          |
| `text`       | Full content of the news article |
| `label`      | Class label: `REAL` or `FAKE`    |

> **Note:** The dataset is not included in this repository. Please download it and place it at:
>
> ```
> data/raw/news.csv
> ```

---

## 🛠️ Project Prerequisites

Install the required libraries with:

```bash
pip install numpy pandas scikit-learn
```

To run the code in a notebook environment, install Jupyter Lab:

```bash
pip install jupyterlab
```

Then launch Jupyter Lab:

```bash
jupyter lab
```

---

## 🚀 Steps for Detecting Fake News

### 1. Make Necessary Imports

```python
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

---

### 2. Load the Dataset

```python
df = pd.read_csv('data/raw/news.csv')
print(df.shape)
df.head()
```

---

### 3. Extract the Labels

```python
labels = df.label
labels.head()
```

---

### 4. Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7
)
```

---

## 📦 Project Structure

```
ml-fake-news-detector/
│
├── data/
│   └── raw/
│       └── news.csv
├── notebooks/
│   └── main_experiment.ipynb
├── main.py
├── requirements.txt
└── README.md
```

---

## 📈 Next Steps

- TF-IDF vectorization of text data
- Train the PassiveAggressiveClassifier
- Evaluate the model with accuracy and confusion matrix
- (Optional) Save the trained model and build an inference API

---

## 📝 License

This project is for educational purposes.
