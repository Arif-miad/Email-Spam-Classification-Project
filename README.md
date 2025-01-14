# Email-Spam-Classification-Project



### Dataset Overview

1. **Dataset Name**
   - Email Spam Classification Dataset

2. **Description**
   - This dataset contains email messages labeled as spam or ham (non-spam).
   - Each entry includes the full content of the email and its classification (0 for ham, 1 for spam).
   - Total observations: 5,728

3. **Columns**
   - `text`: Full content of the email message.
   - `spam`: Binary label indicating whether the email is spam (1) or ham (0).

4. **Dataset Characteristics**
   - **Format**: Tab-delimited (easy to import and process).
   - **Content**: Diverse range of topics and styles typical of real-world emails.
   - **Use Case**: Suitable for natural language processing (NLP), machine learning, and email filtering system development.

5. **Objective**
   - Perform Exploratory Data Analysis (EDA) to understand data distribution, text characteristics, and common patterns in spam and ham emails.
   - Build and evaluate machine learning models (e.g., Logistic Regression, Random Forest) for accurate email classification.

6. **Tasks and Steps**
   - **Data Loading and Inspection**
     - Load dataset into Pandas DataFrame.
     - Check basic statistics and structure.
     - Handle missing values if any.

   - **Exploratory Data Analysis (EDA)**
     - Visualize class distribution (spam vs. ham).
     - Analyze text length distribution for spam and ham emails.
     - Generate word clouds to identify common words in spam and ham emails.
     - Explore common words and n-grams using techniques like TF-IDF.

   - **Data Preprocessing**
     - Clean and preprocess text data (lowercase, remove punctuation, etc.).
     - Convert text data into numerical features suitable for machine learning models.
     - Handle categorical variables (if any).

   - **Model Building and Evaluation**
     - Split dataset into training and testing sets.
     - Implement various classification models (Logistic Regression, SVM, etc.).
     - Evaluate model performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.
     - Compare and select the best-performing model based on evaluation metrics and business requirements.

7. **Conclusion**
   - Summarize findings from EDA and model evaluation.
   - Discuss insights gained and potential improvements for future work.
 



## Overview
This project focuses on classifying emails as either spam (1) or ham (0). Using machine learning and natural language processing (NLP) techniques, we perform exploratory data analysis (EDA) and develop robust models to accurately classify emails.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset Description](#dataset-description)  
3. [Key Objectives](#key-objectives)  
4. [Workflow](#workflow)  
5. [Requirements](#requirements)  
6. [Data Loading and Preprocessing](#data-loading-and-preprocessing)  
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
8. [Feature Engineering](#feature-engineering)  
9. [Model Development](#model-development)  
10. [Performance Metrics and Equations](#performance-metrics-and-equations)  
11. [Results](#results)  
12. [Conclusion](#conclusion)  
13. [References](#references)  

---

## Introduction
Spam emails have become a major concern in the digital era. This project aims to create a robust email classification system using the Email Spam Classification Dataset, which consists of text-based emails and their labels (spam or ham).  

---

## Dataset Description
- **Dataset Name:** Email Spam Classification  
- **Columns:**
  - `text`: Content of the email.
  - `spam`: Label indicating spam (1) or ham (0).  
- **Total Observations:** 5,728 emails  
- **Format:** Tab-delimited  

---

## Key Objectives
1. Perform detailed exploratory data analysis (EDA).  
2. Clean and preprocess the text data.  
3. Engineer features using tokenization, TF-IDF, etc.  
4. Build and evaluate classification models.  
5. Compare models using performance metrics.  

---

## Workflow
### Step-by-Step Implementation:
1. Import necessary libraries.  
2. Load the dataset.  
3. Check for missing values.  
4. Visualize spam vs. ham distribution.  
5. Analyze word frequencies for spam and ham.  
6. Generate word clouds for better insights.  
7. Preprocess text data (lowercase, punctuation removal, etc.).  
8. Remove stopwords and perform lemmatization.  
9. Perform text vectorization using TF-IDF.  
10. Split the dataset into training and testing sets.  
11. Build the following models:
    - Logistic Regression  
    - Naive Bayes  
    - Support Vector Machine (SVM)  
    - Random Forest  
    - Gradient Boosting  
12. Evaluate models using performance metrics:
    - Accuracy  
    - Precision  
    - Recall  
    - F1 Score  
    - ROC and AUC  
13. Perform hyperparameter tuning using GridSearchCV.  
14. Visualize the confusion matrix.  
15. Compare model performance and finalize the best model.  

---

## Requirements
Install the required Python libraries using the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn wordcloud nltk
```

---

## Data Loading and Preprocessing
### Loading Dataset:
```python
# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# Step 2: Load the dataset
data = pd.read_csv("email_spam_dataset.csv")  # Replace with your dataset path
data.columns = ['text', 'spam']  # Ensure column names are correct

# Step 3: Inspect the dataset
print(data.head())
print(data.info())
print(data.describe())

# Step 4: Check for missing values
print("Missing values:\n", data.isnull().sum())

# Step 5: Check class distribution
sns.countplot(data['spam'])
plt.title("Spam vs Ham Distribution")
plt.show()

# Step 6: Convert target column to numeric if not already
data['spam'] = data['spam'].map({'ham': 0, 'spam': 1})  # Adjust based on dataset

# Step 7: Analyze text length
data['text_length'] = data['text'].apply(len)
sns.histplot(data, x='text_length', hue='spam', kde=True)
plt.title("Text Length Distribution")
plt.show()

# Step 8: Word cloud for spam
from wordcloud import WordCloud
spam_words = ' '.join(data[data['spam'] == 1]['text'])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(spam_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Spam Emails")
plt.show()

# Step 9: Word cloud for ham
ham_words = ' '.join(data[data['spam'] == 0]['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Ham Emails")
plt.show()

# Step 10: Analyze most common words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=20)
common_words = cv.fit_transform(data['text'])
common_words_df = pd.DataFrame(cv.get_feature_names_out(), columns=['Word'])
print("Most common words:\n", common_words_df)

# Step 11: Preprocess text (lowercase, remove punctuation, etc.)
import string
import re

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    return text

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Step 12: Split the dataset
X = data['cleaned_text']
y = data['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 13: Transform text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 14: Build logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Step 15: Evaluate logistic regression model
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Step 16: Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)

# Step 17: Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test_tfidf)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Step 18: Support Vector Machine (SVM) model
svm_model = SVC(probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Step 19: Evaluate SVM model
y_pred_svm = svm_model.predict(X_test_tfidf)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Step 20: ROC Curve for Logistic Regression
lr_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, lr_probs)
plt.plot(fpr, tpr, label="Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 21: Compare model performance
models = ['Logistic Regression', 'Random Forest', 'SVM']
accuracies = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svm)]
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

# Step 22: Hyperparameter tuning for Random Forest
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid_rf.fit(X_train_tfidf, y_train)
print("Best Random Forest Parameters:\n", grid_rf.best_params_)

# Step 23: Train tuned Random Forest
rf_tuned = grid_rf.best_estimator_
rf_tuned.fit(X_train_tfidf, y_train)

# Step 24: Evaluate tuned Random Forest
y_pred_rf_tuned = rf_tuned.predict(X_test_tfidf)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))

# Step 25: Save the model and vectorizer
import joblib
joblib.dump(rf_tuned, "spam_classifier_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Step 26: Load saved model
loaded_model = joblib.load("spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Step 27: Test saved model
new_email = ["Congratulations! You've won a prize!"]
new_email_tfidf = loaded_vectorizer.transform(new_email)
print("Prediction for new email:", loaded_model.predict(new_email_tfidf))

# Step 28: Feature importance for Random Forest
feature_importances = rf_tuned.feature_importances_
top_features = np.argsort(feature_importances)[-10:]
plt.barh(np.array(tfidf.get_feature_names_out())[top_features], feature_importances[top_features])
plt.title("Top Features in Random Forest")
plt.show()

# Step 29: Check misclassified emails
misclassified = X_test[(y_test != y_pred_rf_tuned)]
print("Misclassified Emails:\n", misclassified)

# Step 30: Summarize findings
print("Summary: Logistic Regression performed better in terms of [metrics]. Random Forest had higher interpretability due to feature importance.")

```
![](https://github.com/Arif-miad/Email-Spam-Classification-Project/blob/main/e1.PNG)
---

## Performance Metrics and Equations
### Key Metrics:
1. **Accuracy**:
![output](https://github.com/Arif-miad/Email-Spam-Classification-Project/blob/main/eial.PNG)

---

## Results
- **Best Model:** Logistic Regression achieved the highest accuracy (e.g., 95%).
- **Key Insights:**
  - Spam emails often have shorter text lengths.
  - Common spam words include "offer," "free," and "win."  

---

## Conclusion
This project demonstrates a complete workflow for spam classification, starting from data loading to model evaluation. The results show that NLP and machine learning techniques can effectively classify emails as spam or ham.  

---

## References
1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [Kaggle - Spam Email Datasets](https://www.kaggle.com/)
3. [NLTK Library](https://www.nltk.org/)
```

