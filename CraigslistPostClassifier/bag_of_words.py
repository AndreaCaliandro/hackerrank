from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

class BagOfWords():

    def __init__(self, train_docs, train_target, test_docs, test_target,
                 vectorizer=TfidfVectorizer(), classifier=MultinomialNB(alpha=.01)):
        self.train_docs = train_docs
        self.train_target = train_target
        self.test_docs = test_docs
        self.test_target = test_target
        self.vectorizer = vectorizer
        self.classifier = classifier

    def tfidf(self):
        self.train_tfidf = self.vectorizer.fit_transform(self.train_docs)
        self.test_tfidf = self.vectorizer.transform(self.test_docs)

    def modeling(self):
        self.classifier.fit(self.train_tfidf, self.train_target)
        pred = self.classifier.predict(self.test_tfidf)
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
