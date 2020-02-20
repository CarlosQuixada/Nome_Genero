import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

df_train = pd.read_csv('df_train.csv')

text_clf = Pipeline(
    [('vect', CountVectorizer(ngram_range=(1, 1))), ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
     ('clf', LinearSVC())])

text_clf.fit(df_train['nome'].values, df_train['genero'].values)
pickle.dump(text_clf, open("classificador_genero.pkl", 'wb'))
