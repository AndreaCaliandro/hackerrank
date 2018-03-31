
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import metrics


class RunningModel():
    def __init__(self, train_df, test_df, model, target_field, scaling=True):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.model = model
        self.target_field = target_field
        self.scaling = scaling
        self.Features = self.get_features()
        self.Target = self.get_target()

    def get_target(self):
        return self.train_df[self.target_field].values

    def get_features(self):
        return self.train_df.drop([self.target_field], axis=1).as_matrix()

    def training_model(self):
        if self.scaling:
            self.scaler = StandardScaler().fit(self.Features)
            self.X_train = self.scaler.transform(self.Features)
            self.X_test = self.scaler.transform(self.test_df.as_matrix())
        else:
            self.X_train = self.Features
            self.X_test = self.test_df.as_matrix()
        self.model.fit(self.X_train, self.Target)

    def predictions(self):
        return self.model.predict(self.X_test)

    def price_estimator(self):
        y_pred = self.model.predict(self.X_train)
        print 'RMSE on train data:', np.sqrt(metrics.mean_squared_error(self.Target, y_pred))
        self.score(y_pred)

    def score(self, y_pred):
        d = abs(y_pred - self.Target)/self.Target
        vfunc = np.vectorize(lambda x: max(x - 0.1, 0))
        d_adj = vfunc(d)
        scoref = np.vectorize(lambda x: max(1 - x, 0))
        score = scoref(d_adj).mean() * 10
        print 'score = {0:.2f}'.format(score)
