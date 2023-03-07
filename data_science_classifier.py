'''

Data Science Subject Classification Model

Dustin Marshall

Summer Associate, 
Data & Innovation, 
Rockefeller Foundation

September 2022

'''

# Import libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import nltk
import re
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import PrecisionRecallDisplay, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Convert csv with data science proposals text to pandas dataframe & add classification label
data_science_proposals = pd.read_csv("data/DataScienceProposals.csv", header=None) 
data_science_proposals.columns = ["raw_text"]
data_science_proposals["label"] = np.ones(len(data_science_proposals), dtype=np.int32)

# Convert csv with other proposals text to pandas dataframe & add classification label
other_proposals = pd.read_csv("data/OtherProposals.csv", header=None)
other_proposals.columns = ["raw_text"]
other_proposals["label"] = np.zeros(len(other_proposals), dtype=np.int32)

# Print the lenths of both dataframes
print(f"Number of data science proposals : {len(data_science_proposals):,}")
print(f"Number of other proposals : {len(other_proposals):,}")

# Combine dataframed into a single dataframe and shuffle rows
data = pd.concat([data_science_proposals, other_proposals])
data = shuffle(data, random_state=1)

# Preview dataframe
data.head()

# Create function to pre-process text to improve model performance
def text_preprocessing(text):
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # reduce different forms of the same word into their root word
    text = text.split() 
    text = [WordNetLemmatizer().lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# Apply text preprocessing function to each row of raw_text to produce new column
data["processed_text"] = data.apply(lambda row: text_preprocessing(row['raw_text']), axis=1)

# Preview updated dataframe
data.head()

# Find the best classifier
f2_scores_byclassifier = {}
for name, classifier in [
          ('LogReg', LogisticRegression()), 
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC(probability=True)), 
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())]:
    
    # Build pipeline that transforms text and passes through classifier
    model = Pipeline(steps=[
        # tf-idf text vectorization to transform text into frequency-dependent vectors
        ("tfidf", TfidfVectorizer(ngram_range=(2,4), min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english")),
        # perform singular value decomposition to reduce redundancies in the data
        ("svd", TruncatedSVD(n_components=64, n_iter=10, random_state=1)),
        # standardize the data
        ("scaler", StandardScaler()),
        # pass data through classifier
        (name, classifier)])
    
    # Evaluate average model performance across 10 random states (to account for the small sample size)
    f2_scores_byrandomstate = []
    for n in range(1,11):
        # split data into training, validation, and testing sets
        X = data["processed_text"].values
        y = data["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=n, shuffle=True, stratify=y)
        train_X, validate_X, train_y, validate_y = train_test_split(X_train, y_train, test_size=0.25, random_state=n, shuffle=True, stratify=y_train)
        # fit model on the training set
        model.fit(train_X, train_y)
        # find max F2-Score measure for the model
        y_score = model.predict_proba(validate_X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(validate_y, y_score)
        f2_score = (5 * recall * precision) / (4 * recall + precision) + 1e-10
        f2_scores_byrandomstate.append(np.max(f2_score))
    f2_scores_byclassifier[name] = np.average(f2_scores_byrandomstate)

# Print best performing classifier with F2-Score
print('The best performing classifier is', max(f2_scores_byclassifier, key=f2_scores_byclassifier.get),"with an average F2-Score of", max(f2_scores_byclassifier.values()))

# Find the best kernel for an SVM classifier
f2_scores_bykernel = {}
for kernel in ["linear", "rbf", "poly"]:
    model = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(2,4), min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english")),
        ("svd", TruncatedSVD(n_components=64, n_iter=10, random_state=1)),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, probability=True))])
    f2_scores_byrandomstate = []
    for n in range(1,11):
        X = data["processed_text"].values
        y = data["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=n, shuffle=True, stratify=y)
        train_X, validate_X, train_y, validate_y = train_test_split(X_train, y_train, test_size=0.25, random_state=n, shuffle=True, stratify=y_train)
        model.fit(train_X, train_y)
        y_score = model.predict_proba(validate_X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(validate_y, y_score)
        f2_score = (5 * recall * precision) / (4 * recall + precision + 1e-10)
        f2_scores_byrandomstate.append(np.max(f2_score))
    f2_scores_bykernel[kernel] = np.average(f2_scores_byrandomstate)

# Print best performing kernel with F2-Score    
print('The best performing kernel is', max(f2_scores_bykernel, key=f2_scores_bykernel.get),"with an average F2-Score of", max(f2_scores_bykernel.values()))

# Find the best gamma for an SVM classifier with an "rbf" kernel
f2_scores_bygammaval = {}
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(2,4), min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english")),
        ("svd", TruncatedSVD(n_components=64, n_iter=10, random_state=1)),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=gamma, probability=True))])
    f2_scores_byrandomstate = []
    for n in range(1,11):
        # split data into training, validation, and testing sets
        X = data["processed_text"].values
        y = data["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=n, shuffle=True, stratify=y)
        train_X, validate_X, train_y, validate_y = train_test_split(X_train, y_train, test_size=0.25, random_state=n, shuffle=True, stratify=y_train)
        # fit model on the training set
        model.fit(train_X, train_y)
        # find max F2-Score measure for the model
        y_score = model.predict_proba(validate_X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(validate_y, y_score)
        f2_score = (5 * recall * precision) / (4 * recall + precision + 1e-10)
        f2_scores_byrandomstate.append(np.max(f2_score))
    f2_scores_bygammaval[gamma] = np.average(f2_scores_byrandomstate)

# Print best performing gamma with F2-Score      
print('The best performing gamma value is', max(f2_scores_bygammaval, key=f2_scores_bygammaval.get),"with an average F2-Score of", max(f2_scores_bygammaval.values()))

# Find the best C value for an SVM classifier with an "rbf" kernel and gamma value of 0.001
f2_scores_bycval = {}
for c in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(2,4), min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english")),
        ("svd", TruncatedSVD(n_components=64, n_iter=10, random_state=1)),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=0.01, C=c, probability=True))])
    f2_scores_byrandomstate = []
    for n in range(1,11):
        # split data into training, validation, and testing sets
        X = data["processed_text"].values
        y = data["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=n, shuffle=True, stratify=y)
        train_X, validate_X, train_y, validate_y = train_test_split(X_train, y_train, test_size=0.25, random_state=n, shuffle=True, stratify=y_train)
        # fit model on the training set
        model.fit(train_X, train_y)
        # find max F2-Score measure for the model
        y_score = model.predict_proba(validate_X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(validate_y, y_score)
        f2_score = (5 * recall * precision) / (4 * recall + precision) + 1e-10
        f2_scores_byrandomstate.append(np.max(f2_score))
    f2_scores_bycval[c] = np.average(f2_scores_byrandomstate)

# Print best performing C with F2-Score       
print('The best performing C value is ', max(f2_scores_bycval, key=f2_scores_bycval.get)," with an average F2-Score of", max(f2_scores_bycval.values()))    

# Find the average prediction threshold for the best performing model
model = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(ngram_range=(2,4), min_df=5, max_df=0.5, sublinear_tf=True, stop_words="english")),
    ("svd", TruncatedSVD(n_components=64, n_iter=10, random_state=1)),
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", gamma=0.01, C=0.01, probability=True))])
thresholds_byrandomstate = []
for n in range(1,11):
    X = data["processed_text"].values
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=n, shuffle=True, stratify=y)
    train_X, validate_X, train_y, validate_y = train_test_split(X_train, y_train, test_size=0.25, random_state=n, shuffle=True, stratify=y_train)
    model.fit(train_X, train_y)
    y_score = model.predict_proba(validate_X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(validate_y, y_score)
    f2_score = (5 * recall * precision) / (4 * recall + precision + 1e-10)
    thresholds_byrandomstate.append(thresholds[np.argmax(f2_score)])
average_threshold = sum(thresholds_byrandomstate) / len(thresholds_byrandomstate)  

# Print best performing threshold with F2-Score   
print('The average prediction threshold for the best performing model is', average_threshold)    

# Convert csv with data science proposals text to pandas dataframe & add classification label
additional_proposals = pd.read_csv("data/AdditionalProposals.csv", header=None)
additional_proposals.columns = ["raw_text", "year"]

# Preview dataframe
additional_proposals.head()

# Apply text preprocessing function to each row of raw_text to produce new column
additional_proposals["processed_text"] = additional_proposals.apply(lambda row: text_preprocessing(row['raw_text']), axis=1)

# Generate predicted labels for each row of processed test using the model
additional_proposals["predicted_label"] = (model.predict_proba(additional_proposals["processed_text"])[:, 1] > average_threshold).astype('int')

# Preview updated dataframe
additional_proposals.head()

# Plot predicted data science proposals across time
additional_proposals = additional_proposals.groupby("year").sum()
plt.figure(figsize =(20, 10))
plt.bar(additional_proposals.index.values, additional_proposals["predicted_label"])
plt.xlabel("Year", fontsize = 15)
plt.ylabel("Number of Predicted Data Science Proposals", fontsize = 15)
plt.title("Number of Predicted Data Science Proposals by Year", fontweight ='bold', fontsize = 20)
plt.show()