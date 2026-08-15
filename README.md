# 📩 Spam Mail Detector

A machine learning project that classifies textual messages as **Spam** or **Ham (Non-Spam)** using Natural Language Processing (NLP) and machine learning techniques.

## 🎯 Objective

The objective of this project is to build a text classification model that can distinguish between spam and legitimate messages.

The project covers the basic NLP and machine learning pipeline:

**Text Data → Preprocessing → Feature Extraction → Classification → Evaluation**

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**, which contains SMS messages labeled as either:

* **Spam** — unwanted or fraudulent messages
* **Ham** — legitimate/non-spam messages

### Dataset Source

The dataset is publicly available through the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## 🔄 Project Workflow

### 1. Load the Dataset

Load the SMS messages and their corresponding labels into a Pandas DataFrame.

### 2. Text Preprocessing

The text data will be prepared for machine learning by applying:

* Lowercasing
* Tokenization
* Stopword removal
* Basic text cleaning

### 3. Feature Extraction

Convert the processed text into numerical features using:

* Bag of Words (BoW)
* TF-IDF

### 4. Train/Test Split

Split the dataset into training and testing sets to evaluate the model on unseen data.

### 5. Model Training

Train a simple machine learning classification model such as:

* Naive Bayes
* Logistic Regression

### 6. Model Evaluation

Evaluate the classifier using:

* Accuracy
* Precision
* F1-score


## 🚧 Project Status

🟡 **In Progress**

The project is being developed step-by-step, from data loading and preprocessing to model training and evaluation.

## 🎓 Skills Gained

This project focuses on developing practical understanding of:

* Natural Language Processing
* Text preprocessing
* Feature extraction
* TF-IDF and Bag of Words
* Text classification
* Naive Bayes / Logistic Regression
* Model evaluation


