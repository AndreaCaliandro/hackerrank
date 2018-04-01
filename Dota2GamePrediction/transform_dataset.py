import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class TransformDataset():

    def __init__(self, dataset_df, train=True, target_name='winner', counter = CountVectorizer()):
        self.counter = counter
        self.target_name = target_name
        self.dataset_df = dataset_df
        self.data_df = dataset_df.drop(target_name, axis=1) if train else dataset_df

    def get_target(self):
        return self.dataset_df[self.target_name]

    def clean_hero_names(self):
        self.data_df = self.data_df. \
            apply(lambda x: x.str.replace(' ','') if x.dtype == "object" else x). \
            apply(lambda x: x.str.replace("'", '') if x.dtype == "object" else x). \
            apply(lambda x: x.str.replace('-', '') if x.dtype == "object" else x)

    def get_features(self):
        self.clean_hero_names()
        self.counter.fit(self.data_df.apply(lambda x: x+' ').sum(axis=1))
        team_one = self.counter.transform(self.data_df.iloc[:,:5].apply(lambda x: x+' ').sum(axis=1))
        team_two = self.counter.transform(self.data_df.iloc[:,5:].apply(lambda x: x+' ').sum(axis=1))
        return pd.DataFrame(team_one.toarray() - team_two.toarray())
