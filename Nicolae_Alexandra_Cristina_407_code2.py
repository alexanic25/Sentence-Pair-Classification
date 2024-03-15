
!pip install scikit-plot
!pip install scikeras

# importing the necesarry libraries
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import sklearn
import json
import os
import re
from google.colab import drive
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import string
from sklearn import metrics

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

drive.mount('/content/drive', force_remount=True)

# loading the train,test and validation data
path = "/content/drive/MyDrive/data"
train_data = json.load(open(os.path.join(path,'train.json'),'r',encoding='utf-8'))
test_data = json.load(open(os.path.join(path,'test.json'),'r',encoding='utf-8'))
validation_data = json.load(open(os.path.join(path,'validation.json'),'r',encoding='utf-8'))

def get_score(data):
    score = []
    for entry in data:
        score.append(entry['label'])
    return score

y_train = get_score(train_data)
y_validation = get_score(validation_data)

# stop words list from https://www.ranks.nl/stopwords/romanian
stop_words = ['acea','aceasta','această','aceea', 'acei', 'aceia', 'acel', 'acela', 'acele', 'acelea', 'acest', 'acesta', 'aceste', 'acestea', 'acești',
 'aceștia', 'acolo', 'acord', 'acum', 'ai', 'aia', 'aibă', 'aici', 'al', 'ăla', 'ale', 'alea', 'ălea', 'altceva', 'altcineva', 'am', 'ar', 'are', 'aș',
 'așadar', 'asemenea', 'asta', 'ăsta', 'astăzi', 'astea', 'ăstea', 'ăștia', 'asupra', 'aţi', 'au', 'avea', 'avem', 'aveţi', 'azi', 'bine', 'bună', 'ca',
 'că', 'căci', 'când', 'care', 'cărei', 'căror', 'cărui', 'cât', 'câte', 'câţi', 'către', 'câtva', 'caut', 'ce', 'cel', 'ceva', 'chiar','cinci', 'cînd',
 'cine', 'cineva', 'cît', 'cîte', 'cîţi', 'cîtva', 'contra', 'cu','cum', 'cumva', 'curând', 'curînd', 'da', 'dă', 'dacă', 'dar', 'dată', 'datorită', 'dau',
 'de', 'deci', 'deja', 'deoarece', 'departe', 'deși', 'din', 'dinaintea', 'dintr-', 'dintre','doi', 'doilea', 'două', 'drept', 'după', 'ea', 'ei', 'el', 'ele',
 'eram', 'este', 'ești', 'eu', 'face', 'fără', 'fata', 'fi','fie','fiecare', 'fii', 'fim', 'fiţi', 'fiu', 'frumos','graţie', 'halbă', 'iar', 'ieri', 'îi', 'îl',
 'îmi', 'împotriva', 'în', 'înainte', 'înaintea','încât', 'încît', 'încotro', 'între', 'întrucât', 'întrucît', 'îţi', 'la', 'lângă', 'le', 'li','lîngă', 'lor', 'lui',
 'mă', 'mai', 'mâine', 'mea','mei', 'mele', 'mereu', 'meu', 'mi', 'mie', 'mîine', 'mine','mult', 'multă', 'mulţi', 'mulţumesc', 'ne', 'nevoie', 'nicăieri', 'nici',
 'nimeni', 'nimeri', 'nimic', 'niște', 'noastră', 'noastre', 'noi', 'noroc', 'noștri', 'nostru', 'nouă', 'nu', 'opt', 'ori', 'oricând', 'oricare', 'oricât', 'orice',
 'oricînd', 'oricine', 'oricît', 'oricum', 'oriunde', 'până', 'patra', 'patru', 'patrulea', 'pe','pentru', 'peste', 'pic', 'pînă', 'poate', 'pot', 'prea', 'prima',
 'primul', 'prin', 'puţin', 'puţina', 'puţină','rog', 'sa', 'să', 'săi', 'sale', 'șapte','șase', 'sau', 'său', 'se', 'și', 'sînt', 'sîntem', 'sînteţi', 'spate',
 'spre', 'știu', 'sub', 'sunt', 'suntem', 'sunteţi', 'sută', 'ta', 'tăi', 'tale', 'tău', 'te', 'ţi', 'ţie', 'timp', 'tine', 'toată', 'toate', 'tot', 'toţi', 'totuși',
 'trei', 'treia', 'treilea', 'tu', 'un', 'una', 'unde', 'undeva', 'unei', 'uneia','unele', 'uneori', 'unii', 'unor', 'unora', 'unu', 'unui', 'unuia', 'unul', 'vă',
 'vi', 'voastră', 'voastre', 'voi', 'voștri','vostru', 'vouă', 'vreme', 'vreo', 'vreun', 'zece', 'zero', 'zi', 'zice']

# Preprocessing step: applying concatenation of sentences, tokenization and stemming
punct = string.punctuation+'„'+'”'
def concatenate_sentences_and_remove_punctuation(data):
    concatenated_sentences = []
    filtered_sentences = []
    stemmer = SnowballStemmer("romanian")
    for entry in data:
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        concatenated_sentence = f"{sentence1} {sentence2}".lower()
        tokens = word_tokenize(concatenated_sentence)
        filtered_words = [stemmer.stem(word) for word in tokens if word not in punct and word not in stop_words]
        filtered_sentences.append(filtered_words)
    return filtered_sentences

nltk.download('punkt')

updated_train = concatenate_sentences_and_remove_punctuation(train_data)
updated_validation = concatenate_sentences_and_remove_punctuation(validation_data)
updated_test = concatenate_sentences_and_remove_punctuation(test_data)

train_strings = [' '.join(sentence) for sentence in updated_train]
validation_strings = [' '.join(sentence) for sentence in updated_validation]
test_strings = [' '.join(sentence) for sentence in updated_test]

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence, text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Bidirectional, Embedding,Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier

token = text.Tokenizer(num_words=5000)
token.fit_on_texts(train_strings)

# padding the tokenized sentences
def token_pad_sequence(tokenizer, data, max_len):
  return pad_sequences(tokenizer.texts_to_sequences(data), truncating = 'post', padding='post', maxlen=max_len)

train_mat = token_pad_sequence(token,train_strings,20)
validation_mat = token_pad_sequence(token,validation_strings,20)
test_mat = token_pad_sequence(token,test_strings,20)

# Building the model and performing Randomized Search
nn_params = {
    'batch_size':[32, 64],
    'epochs' : [5, 10],
    'optimizer': ['RMSprop', 'Adam']
}

def build_model(optimizer='Adam'):
  model = Sequential()
  model.add(Embedding(5001,output_dim = 200 ,input_length=20))
  model.add(Bidirectional(LSTM(30, return_sequences=True)))
  model.add(Dropout(0.3))
  model.add(Bidirectional(LSTM(30)))
  model.add(Dropout(0.3))
  model.add(Dense(4, activation='softmax'))

  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer ,metrics=['accuracy'])
  return model

nn_classifier = KerasClassifier(build_fn=build_model)
nn_randomized_search = RandomizedSearchCV(estimator=nn_classifier, param_distributions=nn_params, cv=2, verbose = 1)
nn_randomized_search.fit(np.array(train_mat), np.array(y_train))
nn_best_model = nn_randomized_search.best_estimator_

nn_best_model.fit(np.array(train_mat), np.array(y_train))
nn_pred = nn_best_model.predict(np.array(validation_mat))

from sklearn import metrics
print(metrics.classification_report(y_validation,nn_pred))

nn_best_params = nn_randomized_search.best_params_
print('Best params: ', nn_best_params)

print('Accuracy: ', accuracy_score(y_validation,nn_pred))
print(metrics.classification_report(y_validation,nn_pred))

# Concatenating the validation with the train data
X_t_v=np.concatenate([train_mat, validation_mat])

y_t_v = y_train + y_validation

nn_best_model.fit(np.array(X_t_v), np.array(y_t_v),epochs=10, batch_size=32)
nn_pred = nn_best_model.predict(np.array(test_mat))

y_pred = nn_best_model.predict(np.array(test_mat))

test_data_guid = [entry['guid'] for entry in test_data]

# Saving the predicted labels in a CSV
data = {'GUID': test_data_guid,
        'label': y_pred}
df = pd.DataFrame(data)

file_path = '/content/drive/MyDrive/data/submission_34.csv'

df.to_csv(file_path, sep=',', header=['guid', 'label'], index=False)