from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


class BagOfWords():

    def __init__(self, train_docs, train_target,
                 vectorizer=TfidfVectorizer(), classifier=MultinomialNB(alpha=.01)):
        self.train_docs = train_docs
        self.train_target = train_target
        self.vectorizer = vectorizer
        self.classifier = classifier

    def modeling(self):
        self.train_tfidf = self.vectorizer.fit_transform(self.train_docs)
        self.classifier.fit(self.train_tfidf, self.train_target)

    def predict(self, test_docs):
        self.test_tfidf = self.vectorizer.transform(test_docs)
        return self.classifier.predict(self.test_tfidf)

    def estimator(self, test_target, pred):
        score = metrics.accuracy_score(test_target, pred)
        print("accuracy:   %0.3f" % score)

