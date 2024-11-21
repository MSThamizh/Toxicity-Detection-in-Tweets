# **Toxicity Detection in Tweets**

## **Project Overview**
This project implements machine learning models to classify tweets as toxic or non-toxic. The workflow involves advanced **Natural Language Processing (NLP)** techniques for text cleaning, feature extraction using **TF-IDF**, and evaluation of multiple classifiers.
The system processes tweet data, creates robust machine learning models, and evaluates them using metrics like **classification reports**, **confusion matrices**, and **ROC-AUC curves**, ensuring reliable performance in detecting toxicity.

## **Key Features**
1. **Text Preprocessing**:
   - Converts text to lowercase, tokenizes words, removes stopwords, punctuation, and lemmatizes tokens for normalization.

2. **Feature Extraction**:
   - Applies **TF-IDF Vectorization** to represent textual data numerically for machine learning.

3. **Model Training**:
   - Trains classifiers including:
     - Decision Tree
     - Random Forest
     - Naive Bayes
     - K-Nearest Neighbors

4. **Performance Evaluation**:
   - Evaluates models using **classification reports**, **confusion matrices**, and **ROC-AUC** scores for both training and test datasets.

5. **Visualization**:
   - Plots **ROC curves** for assessing model performance visually.

## **Workflow**

### **1. Data Understanding**
The dataset contains:
- **tweet**: Text content of the tweet.
- **Toxicity**: Binary label (1 = toxic, 0 = non-toxic).

### **2. Data Preprocessing**
The raw text data undergoes:
- Lowercasing, punctuation removal, tokenization.
- Stopwords removal using NLTK's English stopword list.
- Lemmatization with NLTK's WordNet lemmatizer.

### **3. Feature Extraction**
Textual data is converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique, enabling machine learning models to process text efficiently.

### **4. Model Training**
The dataset is split into training and test sets. The following models are trained and evaluated:
- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **MultinomialNB**
- **KNeighborsClassifier**

### **5. Model Evaluation**
Each model is evaluated using:
- **Classification Report**: Precision, recall, F1-score for toxic and non-toxic classes.
- **Confusion Matrix**: Visual representation of true/false positives and negatives.
- **ROC Curve & AUC**: Trade-off analysis between true positive and false positive rates.

## **Technologies Used**
- **Python**
- **Libraries**:
  - `pandas` for data manipulation
  - `nltk` for text preprocessing
  - `sklearn` for machine learning and evaluation
  - `matplotlib` for data visualization

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

