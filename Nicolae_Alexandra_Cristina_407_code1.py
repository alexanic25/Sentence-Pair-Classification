!pip install scikit-plot
!pip install xgboost

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
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

# download the punkt package for tokenization
nltk.download('punkt')

drive.mount('/content/drive', force_remount=True)

# loading the train,test and validation data
path = "/content/drive/MyDrive/data"
train_data = json.load(open(os.path.join(path,'train.json'),'r',encoding='utf-8'))
test_data = json.load(open(os.path.join(path,'test.json'),'r',encoding='utf-8'))
validation_data = json.load(open(os.path.join(path,'validation.json'),'r',encoding='utf-8'))

def get_label(data):
    label = []
    for entry in data:
        label.append(entry['label'])
    return label

y_train = get_label(train_data)
y_validation = get_label(validation_data)

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

"""# EDA"""

label0=[]
label1=[]
label2=[]
label3=[]
for entry in train_data:
  sentence1 = entry['sentence1']
  sentence2 = entry['sentence2']
  concatenated_sentence = f"{sentence1} {sentence2}".lower()
  if entry['label'] == 0:
    label0.append(concatenated_sentence)
  elif entry['label'] == 1:
    label1.append(concatenated_sentence)
  elif entry['label'] == 2:
    label2.append(concatenated_sentence)
  else:
    label3.append(concatenated_sentence)

from wordcloud import WordCloud

plt.figure(figsize=(10,8))
wc = WordCloud(background_color="blue", max_words=10000, stopwords=stop_words, max_font_size= 55)
wc.generate(" ".join(label0))
plt.title("Most frequent words from Label0", fontsize=15)
plt.imshow(wc.recolor( colormap= 'Pastel1' , random_state=17))
plt.axis('off')

plt.figure(figsize=(10,8))
wc.generate(" ".join(label1))
plt.title("Most frequent words from Label1", fontsize=15)
plt.imshow(wc.recolor( colormap= 'Pastel1' , random_state=17))
plt.axis('off')

plt.figure(figsize=(10,8))
wc.generate(" ".join(label2))
plt.title("Most frequent words from Label2", fontsize=15)
plt.imshow(wc.recolor( colormap= 'Pastel1' , random_state=17))
plt.axis('off')

plt.figure(figsize=(10,8))
wc.generate(" ".join(label3))
plt.title("Most frequent words from Label3", fontsize=15)
plt.imshow(wc.recolor( colormap= 'Pastel1' , random_state=17))
plt.axis('off')

# Display the relative frequency of labels
y_train_series = pd.Series(y_train)

ax = y_train_series.value_counts(normalize=True).plot.bar()

ax.set_title('Relative Frequency of Labels')
ax.set_xlabel('Labels')
ax.set_ylabel('Relative Frequency')

plt.show()

# Preprocessing step: applying concatenation of sentences, tokenization and stemming
punct = string.punctuation+'„'+'”'

def preprocess_sentences(data):
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

updated_train = preprocess_sentences(train_data)
updated_validation = preprocess_sentences(validation_data)
updated_test = preprocess_sentences(test_data)

train_strings = [' '.join(sentence) for sentence in updated_train]
validation_strings = [' '.join(sentence) for sentence in updated_validation]
test_strings = [' '.join(sentence) for sentence in updated_test]

# filtered sentences to be used in cosine similarity between 2 sentences
def clean_sentences(entry):
    filtered_sentences = []
    stemmer = SnowballStemmer("romanian")
    sentence1 = entry['sentence1']
    sentence2 = entry['sentence2']
    tokens1 = word_tokenize(sentence1)
    tokens2 = word_tokenize(sentence2)
    filtered_words1 = ' '.join([stemmer.stem(word) for word in tokens1 if word not in punct and word not in stop_words])
    filtered_words2 = ' '.join([stemmer.stem(word) for word in tokens2 if word not in punct and word not in stop_words])
    return filtered_words1,filtered_words2

# Computing cosine similarity between 2 sentences
from sklearn.metrics.pairwise import cosine_similarity
def cosine_similarity_tfidf(data,X):
  cosine_list = []
  for entry in data:
    sentences = tfidf.transform(list(clean_sentences(entry)))
    sim = cosine_similarity(sentences)
    cosine_list.append(sim[0][1])
  cosine_list_reshaped = np.array(cosine_list).reshape(-1,1)
  return np.hstack([X.toarray(), cosine_list_reshaped])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3700)
X_train = tfidf.fit_transform(train_strings)
X_validation = tfidf.transform(validation_strings)
X_test = tfidf.transform(test_strings)

X_train_with_cosine_similarity = cosine_similarity_tfidf(train_data,X_train)
X_validation_with_cosine_similarity = cosine_similarity_tfidf(validation_data,X_validation)
X_test_with_cosine_similarity = cosine_similarity_tfidf(test_data,X_test)

from sklearn.naive_bayes import MultinomialNB

# Define parameters for Multinomial Naive Bayes
nb_params = {
    'alpha': [0.5, 0.9, 1.0, 1.5]
}


# Perform grid search
nb_model = MultinomialNB()
nb_grid_search = GridSearchCV(nb_model, nb_params, scoring='f1_macro', verbose=1)
nb_grid_search.fit(X_train_with_cosine_similarity, y_train)

nb_best_model = nb_grid_search.best_estimator_

nb_best_model.fit(X_train_with_cosine_similarity, y_train)
nb_pred = nb_best_model.predict(X_validation_with_cosine_similarity)
nb_best_params = nb_grid_search.best_params_
print('Best params: ', nb_best_params)

from sklearn import metrics
print(metrics.classification_report(y_validation,nb_pred))

skplt.metrics.plot_confusion_matrix(y_validation,nb_pred,normalize=True)

from sklearn.ensemble import RandomForestClassifier

# Define parameters for Random Forest
rf_params = {
    'n_estimators': [50,100]
}

# Perform grid search
rf_model = RandomForestClassifier()
rf_grid_search = GridSearchCV(rf_model, rf_params, scoring='f1_macro', cv = 2, verbose=1)
rf_grid_search.fit(X_train_with_cosine_similarity, y_train)

rf_best_model = rf_grid_search.best_estimator_

rf_best_model.fit(X_train_with_cosine_similarity, y_train)
rf_pred = rf_best_model.predict(X_validation_with_cosine_similarity)
rf_best_params = rf_grid_search.best_params_
print('Best params: ', rf_best_params)

from sklearn import metrics
print(metrics.classification_report(y_validation,rf_pred))

skplt.metrics.plot_confusion_matrix(y_validation,rf_pred,normalize=True)

import xgboost as xgb
xgb=xgb.XGBClassifier(n_estimators = 100,learning_rate = 0.1)
xgb.fit(X_train_with_cosine_similarity, y_train)
y_pred = xgb.predict(X_validation_with_cosine_similarity)
print(accuracy_score(y_validation,y_pred))
print(metrics.classification_report(y_validation,y_pred))

skplt.metrics.plot_confusion_matrix(y_validation,y_pred,normalize=True)

# Table with the ML models
all = pd.DataFrame({'Model name': ['Multinomial Naive Bayes','Random Forest Classifier','Extreme Gradient Boosting Classifier'],
                    'Accuracy':[round(accuracy_score(y_validation,nb_pred),4) ,round(accuracy_score(y_validation,rf_pred),4),round(accuracy_score(y_validation,y_pred),4)],
                    'F1 macro':[round(f1_score(y_validation,nb_pred, average='macro'),4),round(f1_score(y_validation,rf_pred, average='macro'),4),round(f1_score(y_validation,y_pred, average='macro'),4)]})
all.sort_values(by = 'F1 macro',ascending = False).reset_index(drop=True)

# Concatenating the validation with the train data
X_t_v=np.concatenate([X_train_with_cosine_similarity, X_validation_with_cosine_similarity])
y_t_v=y_train+y_validation

#nb_best_model.fit(X_t_v, y_t_v)
#y_pred_test = nb_best_model.predict(X_test_with_cosine_similarity)

#xgb.fit(X_t_v, y_t_v)
#y_pred_test = xgb.predict(X_test_with_cosine_similarity)

rf_best_model.fit(X_t_v, y_t_v)

y_pred_test = rf_best_model.predict(X_test_with_cosine_similarity)
y_pred_test

test_data_guid = [entry['guid'] for entry in test_data]

# Saving the predicted labels in a CSV
data = {'GUID': test_data_guid,
        'label': y_pred_test}
df = pd.DataFrame(data)

file_path = '/content/drive/MyDrive/data/submission_32.csv'

df.to_csv(file_path, sep=',', header=['guid', 'label'], index=False)


"""# LDA"""

# from sklearn import decomposition
# lda = decomposition.LatentDirichletAllocation(n_components=5, max_iter=5,
#                                 learning_method = 'online',
#                                 learning_offset = 50.,
#                                 random_state = 0)

# lda.fit(X_train)
# lda_topic_distribution = lda.transform(X_train)


# lda_topic_distribution_validation = lda.transform(X_validation)

# from scipy.sparse import hstack
# X_validation_with_topics = hstack([X_validation, lda_topic_distribution_validation])

# # Concatenate topic distribution with original features
# X_train_with_topics = hstack([X_train, lda_topic_distribution])

# lda_topic_distribution_test = lda.transform(X_test)
# X_test_with_topics = hstack([X_test, lda_topic_distribution_test])

# #varianta cu 200 clase
# from sklearn.ensemble import RandomForestClassifier
# rf=RandomForestClassifier(n_estimators = 100,random_state = 42)
# rf.fit(X_train_with_topics, y_train)
# y_pred = rf.predict(X_validation_with_topics)
# print(accuracy_score(y_validation,y_pred))
