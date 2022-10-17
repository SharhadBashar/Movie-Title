import pickle
import numpy as np
import pandas as pd
from cleantext import clean

import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")

from googletrans import Translator

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, FeatureHasher, TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

import warnings
warnings.filterwarnings('ignore')



class Clean:
	def __init__(self, csv_filename = 'movies.csv', csv_cleaned_filename = 'movies_cleaned.csv', excel_filename = 'VMR Python Data TEST Sep\'22.xlsx', input_type = 'csv'):
		if (input_type == 'excel'):
			self.convert_to_csv(excel_filename)
		df = read_csv(csv_filename)
		df_cleaned = self.clean_data(df)
		self.save_df(df_cleaned, csv_cleaned_filename)

	def convert_to_csv(self, filename = 'VMR Python Data TEST Sep\'22.xlsx'):
	    df = pd.DataFrame(pd.read_excel(filename))
	    df.to_csv('movies.csv', index = None, header = ['name', 'class'])

	def read_csv(self, filename = 'movies.csv'):
	    return pd.read_csv(filename)

	def clean_data(self, df, translate = False, stem = False, lemm = True):
	    translator = Translator()
	    stop = stopwords.words('english')
	    stemmer = PorterStemmer()
	    lemmatizer = WordNetLemmatizer()
	    
	    df['name'] = df['name'].str.replace('[^A-Za-z0-9 ]+', ' ')
	    if translate: 
	        df['name'] = df['name'].apply(lambda x: translator.translate(x, dest = 'en'))
	    df['name'] = df['name'].apply(lambda x: clean(x))
	    df['name'] = df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	    if stem:
	        df['name'] = df['name'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
	    if lemm:
	        df['name'] = df['name'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
	    
	    df['class'] = df['class'].map({'Entertainment': 0, 'News': 1, 'Sports': 2})
	    df = df.dropna()
	    return df

	def save_df(self, df, filename = 'movies_cleaned.csv'):
	    df.to_csv(filename, index = None)

class Train:
	def __init__(self, csv_filename = 'movies.csv'):
		X, y = self.get_data(csv_filename)
		X = self.bag_of_words(X)
		model = self.train(X, y, LogisticRegression(C = 100.0, random_state = 1, solver = 'lbfgs', multi_class = 'ovr'))
		self.save_model(model)

	def get_data(self, filename = 'movies_cleaned.csv'):
	    df = shuffle(pd.read_csv(filename).dropna())
	    X = df['name']
	    y = df['class']
	    return X, y

	def bag_of_words(self, X):
	    vectorizer = CountVectorizer(stop_words = 'english')
	    X = vectorizer.fit_transform(X)
	    return X

	def one_hot_encoding(self, X):
	    one_hot_encoder = OneHotEncoder()
	    X = X.values.reshape(-1, 1)
	    X = one_hot_encoder.fit_transform(X)
	    return X

	def word_2_vector(self, X):
	    w2v_model = gensim.models.Word2Vec(X, vector_size = 100, window = 5, min_count = 2)

	def glove(self, X):
	    return X

	def tfidf(self, X):
	    tfidf_vectorizer = TfidfVectorizer(max_df = 0.8, max_features = 10000)
	    X = tfidf_vectorizer.fit_transform(X)
	    return X

	def train(self, X, y, model):
		if model is None:
			model = LogisticRegression(C = 100.0, random_state = 1, solver = 'lbfgs', multi_class = 'ovr')
		model.fit(X, y)
		return model

	def save_model(self, model, filename = 'tained_model.pkl'):
		pickle.dump(model, open(filename, 'wb'))

class Predict:
	def __init__(self, csv_filename = 'movies.csv', csv_predicted_filename = 'movies_predicted.csv', excel_filename = 'VMR Python Data TEST Sep\'22.xlsx', input_type = 'excel'):
		if (input_type == 'excel'):
			self.convert_to_csv(excel_filename)
		df = read_csv(csv_filename)
		df_cleaned = self.clean_data(df)
		df_predict = self.predict(df_cleaned)
		self.save_df(df_predict, csv_predicted_filename)

	def convert_to_csv(self, filename = 'VMR Python Data TEST Sep\'22.xlsx'):
	    df = pd.DataFrame(pd.read_excel(filename))
	    df.to_csv('movies.csv', index = None, header = ['name'])

	def read_csv(self, filename = 'movies.csv'):
	    return pd.read_csv(filename)

	def clean_data(self, df, translate = False, stem = False, lemm = True):
	    translator = Translator()
	    stop = stopwords.words('english')
	    stemmer = PorterStemmer()
	    lemmatizer = WordNetLemmatizer()
	    
	    df['name'] = df['name'].str.replace('[^A-Za-z0-9 ]+', ' ')
	    if translate: 
	        df['name'] = df['name'].apply(lambda x: translator.translate(x, dest = 'en'))
	    df['name'] = df['name'].apply(lambda x: clean(x))
	    df['name'] = df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	    if stem:
	        df['name'] = df['name'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
	    if lemm:
	        df['name'] = df['name'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
	    
	    df = df.dropna()
	    return df

	def predict(self, df, filename = 'tained_model.pkl'):
		model = pickle.load(open(filename, 'rb'))
		df['class'] = model.predict(df['name'].values)
		return df

	def save_df(self, df, filename = 'movies_cleaned.csv'):
	    df.to_csv(filename, index = None)		

    
    










