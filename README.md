# **Toxicity Detection in Tweets**

This project focuses on detecting toxic tweets using advanced Natural Language Processing (NLP) techniques. The goal is to classify tweets as either toxic or non-toxic, based on the content, to help prevent harmful online communication.

## **Problem Statement**
The task is to build a model that classifies tweets into two categories: toxic and non-toxic. Toxicity is defined as language that is harmful, abusive, or hateful. The model is trained using a dataset of labeled tweets and evaluated using classification metrics.

## Workflow

1. **Data Collection**: The dataset consists of labeled tweets, where each tweet is classified as either toxic or non-toxic.
2. **Data Preprocessing**: Text data is cleaned and preprocessed by:
   - Removing special characters, URLs, and stopwords.
   - Tokenizing text and lemmatizing words.
3. **Feature Extraction**: 
   - Used **TF-IDF vectorization** to convert the text data into numerical features for the machine learning models.
4. **Model Training**: 
   - Trained multiple machine learning classifiers like **Logistic Regression**, **Random Forest**, and **SVM**.
   - Hyperparameter tuning was performed for better model accuracy.
5. **Model Evaluation**: 
   - Evaluated models using metrics such as **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.
6. **Prediction**: 
   - The model predicts whether a given tweet is toxic or non-toxic.

## Features

- **Text Preprocessing**: 
  - Removal of special characters, URLs, and non-alphabetic characters.
  - Tokenization and lemmatization for better feature extraction.
  
- **TF-IDF Vectorization**: 
  - Convert text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
  
- **Model Training & Evaluation**:
  - Multiple models are trained, evaluated, and compared based on classification metrics.

- **User Interface**: 
  - User can input a tweet and the model will classify it as toxic or non-toxic.
  
## Technologies Used

- **Python**: Primary language for model development.
- **NLP Libraries**: 
  - **NLTK** for text preprocessing (tokenization, lemmatization).
  - **Scikit-learn** for machine learning algorithms and evaluation.
- **TF-IDF**: For converting text into numerical features.


## **Results**

### **Classification Models**
| Model                  | Test Accuracy | Precision (Toxic) | Recall (Toxic) | F1-Score (Toxic) | Key Insights                              |
|-------------------------|---------------|--------------------|----------------|------------------|-------------------------------------------|
| Decision Tree           | 87%           | 80%               | 91%            | 85%             | High interpretability; prone to overfitting. |
| Random Forest           | 91%           | 88%               | 90%            | 89%             | Robust to overfitting; high accuracy.       |
| Multinomial Naive Bayes | 89%           | 87%               | 87%            | 87%             | Performs well with text data.              |
| K-Nearest Neighbors     | 81%           | 71%               | 91%            | 80%             | Struggles with imbalanced classes.         |

### **Visualizations**
- **ROC Curves**:
  Each model's ROC curve shows its capability to distinguish between toxic and non-toxic tweets.
- **Confusion Matrices**:
  Evaluate the model's ability to avoid false positives and false negatives.

## References

- **Python**: [https://docs.python.org/3/](https://docs.python.org/3/)
- **NLTK**: [https://www.nltk.org/](https://www.nltk.org/)
- **Scikit-learn Documentation**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

