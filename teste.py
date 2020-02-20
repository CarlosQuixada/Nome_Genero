import pickle

import pandas as pd
from sklearn import metrics

df_test = pd.read_csv('df_test.csv')

text_clf = pickle.load(open("classificador_genero.pkl", 'rb'))

pred = text_clf.predict(df_test['nome'].values)

precision = metrics.precision_score(df_test['genero'].values, pred, average='macro')
recall = metrics.recall_score(df_test['genero'].values, pred, average='macro')

print("##########")
print(text_clf.classes_)

print(df_test['nome'].values)
print(pred)
print("Acuracia: " + str(metrics.accuracy_score(df_test['genero'].values, pred)))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1-Score: " + str(2 * ((precision * recall) / (precision + recall))))
