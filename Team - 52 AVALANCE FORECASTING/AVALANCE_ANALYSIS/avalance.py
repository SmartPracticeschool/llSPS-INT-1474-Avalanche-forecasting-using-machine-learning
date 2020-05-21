# Created by @DD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Avalance_datasets.csv')

dataset['test_heat'].fillna(0, inplace=True)

dataset['stickiness_score'].fillna(dataset['stickiness_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['test_heat'] = X['test_heat'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('avalance.pkl','wb'))

model = pickle.load(open('avalance.pkl','rb'))
print(model.predict([[2, 1, 1]]))