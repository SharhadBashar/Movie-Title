import pandas as pd
# from googletrans import Translator
from cleantext import clean

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

class Data:
    def __init__(self, 
                 excel_filename = 'VMR Python Data TEST Sep\'22.xlsx', 
                 csv_filename = 'movies.csv',
                 train = True,
                 convert = False, 
                 translate = False, 
                 stem = False, 
                 lemm = True):
        if convert: self.convert_to_csv(csv_filename = csv_filename, 
                                        excel_filename = excel_filename,
                                        train = train)
        df = self.read_csv(filename = csv_filename)
        df = self.clean_data(df, translate = False, stem = False, lemm = True, train = train)
        self.save_df(df, filename = csv_filename)
        
    def convert_to_csv(self, csv_filename = 'movies.csv', 
                       excel_filename = 'VMR Python Data TEST Sep\'22.xlsx',
                       train = True):
        header = ['name']
        if train:
            header = ['name', 'class']
        df = pd.DataFrame(pd.read_excel(excel_filename))
        df.to_csv(csv_filename, index = None, header = header)

    def read_csv(self, filename = 'movies.csv'):
        return pd.read_csv(filename)

    def clean_data(self, df, translate = False, stem = False, lemm = True, train = True):
#         translator = Translator()
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

        if train:
            df['class'] = df['class'].map({'Entertainment': 0, 'News': 1, 'Sports': 2})
        df = df.dropna()
        return df

    def save_df(self, df, filename = 'movies.csv'):
        df.to_csv(filename, index = None)
        print('Data has been cleaned and saved at {}'.format(filename))