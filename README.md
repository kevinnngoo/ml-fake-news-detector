# 📰 Fake News Detection with Python and Machine Learning

This project demonstrates how to build a machine learning pipeline to classify news articles as **REAL** or **FAKE** using Python, pandas, and scikit-learn. The workflow includes data loading, preprocessing, feature extraction, model training, and evaluation.

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
jupyter lab
```

---

## 🚀 Steps for Detecting Fake News

1. **Import Libraries**

   ```python
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import PassiveAggressiveClassifier
   from sklearn.metrics import accuracy_score, confusion_matrix
   ```

2. **Load the Data**

   ```python
   df = pd.read_csv('data/raw/news.csv')
   print(df.shape)
   df.head()
   ```

3. **Extract Labels**

   ```python
   labels = df.label
   labels.head()
   ```

4. **Split the Dataset**

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       df['text'], labels, test_size=0.2, random_state=7
   )
   ```

5. **TF-IDF Vectorization**

   ```python
   tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
   tfidf_train = tfidf_vectorizer.fit_transform(X_train)
   tfidf_test = tfidf_vectorizer.transform(X_test)
   ```

6. **Train the Model**

   ```python
   pac = PassiveAggressiveClassifier(max_iter=50)
   pac.fit(tfidf_train, y_train)
   y_pred = pac.predict(tfidf_test)
   score = accuracy_score(y_test, y_pred)
   print(f'Accuracy: {round(score*100,2)}%')
   ```

7. **Evaluate with Confusion Matrix**
   ```python
   print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))
   ```

---

## 🏆 Results

- **Accuracy achieved:** 92.82%
- **Confusion Matrix:**
  - 589 true positives
  - 587 true negatives
  - 42 false positives
  - 49 false negatives

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

## 📝 Summary

I built a fake news detector using a political news dataset, TF-IDF vectorization, and a PassiveAggressiveClassifier. The model achieved over 92% accuracy in classifying news as REAL or FAKE.

---

## 📄 License

This project is for educational purposes.

## ▶️ Usage

Open `notebooks/main_experiment.ipynb` in Jupyter Lab and run all cells to reproduce the results.
