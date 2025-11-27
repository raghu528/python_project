🛡️ Spam Detection & Fake Review Identification

A Machine Learning project for detecting spam and deceptive online reviews using supervised and semi-supervised learning techniques.

This project analyzes text reviews, extracts linguistic + behavioral features, and classifies them as spam or genuine.

🔍 Overview

This system detects fake/spam reviews by analyzing:

Review content

Reviewer behavior

Linguistic & psycholinguistic patterns

Word frequencies, sentiment, and review length

The model is trained using both labeled and unlabeled data to improve accuracy when labeled data is limited.

🧠 Techniques Used
▶️ Supervised Learning

Naive Bayes

Support Vector Machine (SVM)

▶️ Semi-Supervised Learning

Expectation Maximization (EM)

Positive-Unlabeled Learning

Co-training, Label Propagation (referenced)

▶️ NLP Processing

Tokenization

Stop-word removal

Stemming (Porter Stemmer)

Bag-of-Words

Sentiment polarity

Word frequency vectors

Review length extraction

📊 Dataset

Labeled dataset from previous research (Ott et al.)

Balanced set of spam & truthful opinions

Preprocessing removes:

Handles (@user)

Special characters

Short words

Noise

🚀 Model Performance
Model Type	Algorithm	Accuracy
Supervised	Naive Bayes	86.32%
Supervised	SVM	82%
Semi-Supervised	EM + NB	85.21%
Semi-Supervised	EM + SVM	81%
🛠️ Tech Stack

Python

Scikit-learn

NumPy

Matplotlib / Seaborn

NLTK

WordCloud

Spyder / Anaconda

📌 Features

✔️ Detect fake reviews using text classification
✔️ Extract behavioral & linguistic features
✔️ Supports supervised + semi-supervised learning
✔️ Visual analysis (WordCloud, plots)
✔️ High accuracy with Naive Bayes + EM
✔️ Modular, customizable, extendable

🧩 System Architecture

Main pipeline stages:

Data Preprocessing

Feature Extraction (Word freq, sentiment, length)

Train/Test Split

Classification (NB, SVM, EM)

Prediction

Evaluation (Confusion matrix, accuracy)

📁 Project Structure
Spam-Detection/
│── data/
│── preprocessing/
│── models/
│── feature_extraction/
│── results/
│── spam_detection.py
│── requirements.txt
│── README.md

🧪 How to Run
1️⃣ Install Dependencies
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud

2️⃣ Run Preprocessing
python spam_detection.py

3️⃣ Train Models

Automatically runs NB, SVM & EM-based approaches.

4️⃣ View Results

Accuracy, confusion matrix & WordClouds are generated.

📸 Visualizations

You can include:

WordCloud of genuine reviews

WordCloud of spam reviews

Accuracy comparison bar chart

(Add screenshots if available)

🔮 Future Enhancements

Deep Learning (LSTM/CNN) for text classification

Multi-language spam detection

Use POS tagging & dependency parsing

Real-time spam detection API

Reviewer behavioral graph analysis
