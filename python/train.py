import os
import pickle
import pandas as pd

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data import Data

class Train:
    def __init__(self, 
                 csv_filename = 'movies.csv',
                 excel_filename = 'VMR Python Data TEST Sep\'22.xlsx',
                 delete_csv = False,
                 delete_excel = False,
                 model_filename = 'model.pkl'):
        self.train(csv_filename = 'movies.csv', model_filename = 'model.pkl')
        if delete_csv:
            os.remove(csv_filename)
        if delete_excel:
            os.remove(excel_filename)
        
    def get_data(self, 
                 excel_filename = 'VMR Python Data TEST Sep\'22.xlsx',
                 csv_filename = 'movies.csv'):
        Data(excel_filename = 'VMR Python Data TEST Sep\'22.xlsx', 
             csv_filename = 'movies.csv',
            convert = True)
        df = shuffle(pd.read_csv(csv_filename).dropna())
        X = df['name']
        y = df['class']
        return X, y
    
    def train(self, csv_filename = 'movies.csv', model_filename = 'model.pkl'):
        X, y = self.get_data(csv_filename = csv_filename)
        movie_clf = Pipeline([
             ('vect', CountVectorizer(stop_words = 'english')),
             ('tfidf', TfidfTransformer()),
             ('clf', LogisticRegression(C = 100.0, random_state = 1, solver = 'lbfgs', multi_class = 'ovr'))
        ])
        model = movie_clf.fit(X, y)
        pickle.dump(model, open(model_filename, 'wb'))
        print('Model saved at {}'.format(model_filename))