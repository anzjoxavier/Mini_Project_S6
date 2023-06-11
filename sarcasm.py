import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_json("E:\ANZ JO\Mini Project\Sarcasm Analyser\Sarcasm_Headlines_Dataset.json",lines=True)
print("Dimensions of the data")
# print(data.shape)
# print("Displaying the first 5 rows of the dataset")
# print(data.head())
# Intial data cleaning
# print(data.duplicated().sum())
#Removing the duplicates
data.drop_duplicates(inplace=True)
# print(data.duplicated().sum())
# print("Displaying the count of sarcastic headlines")

# print("Displaying the attributes in Datasets")
# print(data.dtypes)
print("Article Link column is not required for buliding machine learning model.")
data.drop(columns=['article_link'], inplace=True)
data["headline"] = data["headline"].str.lower()
# print(data.head())
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

data["headline"] = data["headline"].apply(lambda text: remove_punctuation(text))
# print(data.head())


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

data["headline"] = data["headline"].apply(lambda text: remove_stopwords(text))

# print(data.head())
cnt = Counter()
for text in data["headline"].values:
    for word in text.split():
        cnt[word] += 1       
# print(cnt.most_common(10))

FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

data["headline"] = data["headline"].apply(lambda text: remove_freqwords(text))
# print(data.head())

n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

data["headline"] = data["headline"].apply(lambda text: remove_rarewords(text))
# print(data.head())


def preprocess_signs(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub('\s+', ' ', text).strip()

    return text

data['headline'] = data['headline'].apply(preprocess_signs)
# print(data.head())

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['headline'])
y = data['is_sarcastic']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# # Xtest=vectorizer.fit_transform(X_test)
# # ytest=y_test

# clf = LogisticRegression()
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# print(y_pred)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy:", acc)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("Shape of :")
# print(X_train)
# print(y_train)
# vectorizer = TfidfVectorizer()

# X = vectorizer.fit_transform(X_train)
# y = y_train
clf = LogisticRegression()
clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# vectorizer1 = TfidfVectorizer()
# XTest = vectorizer1.fit_transform(X_test)
# yTest = y_test
# y_pred = clf.predict(XTest)
# acc = accuracy_score(yTest, y_pred)
# X = vectorizer.fit_transform(X_test)
# y = y_test
# print(data['headline'][0])
tfidf2 = TfidfVectorizer()
f=tfidf2.fit_transform([data['headline'][0],data['headline'][1]])
print(f)
# print(X_test[0])
y_pred = clf.predict(f)
print(y_pred)











