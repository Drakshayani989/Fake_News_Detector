# 📰 Fake News Detector

A Machine Learning-powered web application that classifies news articles as **Fake News** or **Real News** using Natural Language Processing (NLP), TF-IDF Vectorization, and Logistic Regression.

## 🚀 Live Project Overview

This project analyzes the content of a news article and predicts whether it is **Fake** or **Real** based on patterns learned from thousands of labeled news articles.

The application is built with:

- Python
- Scikit-Learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

---

## 📌 Features

✅ Fake vs Real News Classification

✅ Interactive Streamlit Dashboard

✅ Confidence Score Display

✅ Word & Character Count Analysis

✅ Clean Recruiter-Friendly UI

✅ Trained on 44,000+ News Articles

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| NLP | TF-IDF Vectorization |
| Model | Logistic Regression |
| Web App | Streamlit |
| Model Storage | Joblib |

---

## 📂 Dataset

**Dataset Used:** ISOT Fake and Real News Dataset

The dataset contains thousands of news articles labeled as:

- Fake News (0)
- Real News (1)

Dataset Source:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

> Dataset files are not included in this repository due to GitHub storage limitations.

---

## 🧠 Machine Learning Pipeline

### 1. Data Collection

- Loaded Fake.csv
- Loaded True.csv

### 2. Data Preprocessing

- Assigned labels
- Combined title and article text
- Merged datasets
- Shuffled data

### 3. Feature Engineering

Used **TF-IDF Vectorization** to convert textual data into numerical features.

### 4. Model Training

Algorithm Used:

- Logistic Regression

### 5. Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📊 Results

| Metric | Score |
|----------|----------|
| Accuracy | 98.56% |
| Precision | 99% |
| Recall | 99% |
| F1 Score | 99% |

### Confusion Matrix

```text
[[4629   81]
 [  48 4222]]
```

---

## 📁 Project Structure

```text
Fake-News-Detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── Fake_News_Detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Drakshayani989/Fake_News_Detector.git
cd Fake_News_Detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## ⚠️ Limitations

- The model performs best on complete news articles.
- Predictions for short headlines may be less reliable.
- The model identifies linguistic patterns and does not verify facts from the internet.
- Recent news events may sometimes be misclassified because the dataset contains historical news articles.

---

## 🔮 Future Improvements

- BERT-Based Fake News Detection
- Explainable AI (XAI)
- News Source Verification
- Real-Time News API Integration
- Advanced Analytics Dashboard

---

## 👨‍💻 Author

Developed by **Drakshayani Kinjarapu**

Machine Learning | NLP | Python

---

### ⭐ If you found this project useful, consider giving it a star.
