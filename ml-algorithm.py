import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier
import requests
from pyquery import PyQuery as pq

#scraping!!
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0'}
url = str(input("Enter a url: "))
type(url)
response = requests.get(url, headers = headers)
htmlstring = response.content
doc = pq(htmlstring)
heading = (doc('h1').text())
title = (doc('h2').text())


#importing the data
df = pd.read_csv("C:\Users\dhany\Desktop\hfake_or_real_news.csv")
print df.shape
df.head()
df = df.set_index("Unnamed: 0")

d = {'': [''], 'title': [heading], 'text': [title], 'label':['']}
bf = pd.DataFrame(data = d)
print bf.shape


# Set `y`
y = df.label

# Drop the `label` column
df.drop("label", axis=1)

# Make training and test set
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words = 'english')
count_train = count_vectorizer.fit_transform(X_train)
print X_train.head()
count_test = count_vectorizer.transform(X_test)
count_blind = count_vectorizer.transform(bf)


tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
tfidf_blind = tfidf_vectorizer.transform(bf)

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    #print(cm)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment = "center",
            color = "white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


clf = MultinomialNB()
clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
#print("accuracy: %0.3f" % score)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


linear_clf = PassiveAggressiveClassifier(n_iter = 50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
#print("accuracy: %0.3f" % score)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

clf = MultinomialNB(alpha=0.1)
last_score = 0
for alpha in np.arange(0, 1, .1):
    nb_classifier = MultinomialNB(alpha = alpha)
    nb_classifier.fit(tfidf_train, y_train)
    pred = nb_classifier.predict(tfidf_test)
    score = accuracy_score(y_test, pred)
    if score > last_score:
        clf = nb_classifier
    print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)

most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)

# feature_names = tfidf_vectorizer.get_feature_names()
print()
print clf.predict(tfidf_blind)

