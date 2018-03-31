from sklearn.feature_extraction.text import TfidfVectorizer

class BagOfWords():

    def __init__(self, train_docs, test_docs, vectorizer=TfidfVectorizer()):
        self.train_docs = train_docs
        self.test_docs = test_docs
        self.vectorizer = vectorizer

    def tfidf(self):
        self.train_tfidf = self.vectorizer.fit_transform(self.train_docs)
        self.test_tfidf = self.vectorizer.transform(self.test_docs)
