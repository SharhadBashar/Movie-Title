import os
import pickle
import pandas as pd

from data import Data

class Predict:
    def __init__(self, 
                 csv_filename = 'movies.csv',
                 out_filename = 'movies_predict.csv',
                 excel_filename = 'VMR Python Data TEST Sep\'22_predict.xlsx',
                 model_filename = 'model.pkl',
                 delete_csv = False,
                 delete_excel = False):
        self.predict(csv_filename = 'movies.csv',
                     excel_filename = 'VMR Python Data TEST Sep\'22_predict.xlsx',
                     model_filename = 'model.pkl',
                     out_filename = 'movies_predict.csv')
        if delete_csv:
            os.remove(csv_filename)
        if delete_excel:
            os.remove(excel_filename)
        
    def get_data(self, 
                 excel_filename = 'VMR Python Data TEST Sep\'22_predict.xlsx', 
                 csv_filename = 'movies.csv'):
        Data(excel_filename = 'VMR Python Data TEST Sep\'22_predict.xlsx', 
             csv_filename = 'movies.csv', 
             convert = True, train = False)
        df = pd.read_csv(csv_filename).dropna()
        X = df['name']
        return X
    
    def predict(self, 
                csv_filename = 'movies.csv',
                excel_filename = 'VMR Python Data TEST Sep\'22_predict.xlsx',
                model_filename = 'model.pkl',
                out_filename = 'movies_predict.csv'):
        X = self.get_data(csv_filename = 'movies.csv',
                 excel_filename = 'VMR Python Data TEST Sep\'22_predict.xlsx')
        model = pickle.load(open(model_filename, 'rb'))
        y_predict = model.predict(X)
        pd.DataFrame({'name': X, 'class': y_predict}).to_csv(out_filename, index = False)
        print('Predictions saved at {}'.format(out_filename))