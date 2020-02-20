import pandas as pd

from sklearn.model_selection import train_test_split
from preprocessing.text.processing import PreProcessing

process = PreProcessing()

df = pd.read_csv('brazilian-names.csv')

nomes = df['Nome'].values
generos = df['Genero'].values

nome_train, nome_test, genero_train, genero_test = train_test_split(nomes, generos, test_size=0.20, stratify=generos)

df_train = pd.DataFrame({
    'nome': nome_train,
    'genero': genero_train
})

df_test = pd.DataFrame({
    'nome': nome_test,
    'genero': genero_test
})


def preprocess(text):
    text = text.lower()
    text = process.remove_accent(text)

    return text


df_train['nome'] = df_train['nome'].apply(preprocess)
df_train['genero'] = df_train['genero'].apply(str.lower)

df_test['nome'] = df_train['nome'].apply(preprocess)
df_test['genero'] = df_train['genero'].apply(str.lower)

df_train.to_csv('df_train.csv', index=False)
df_test.to_csv('df_test.csv', index=False)
