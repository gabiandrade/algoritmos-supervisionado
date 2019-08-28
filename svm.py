## Deteção de insultos racistas no Twitter

import glob
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

pos_files = glob.glob("C:/Users/Gabriela/Desktop/classificacao/positivos/*.txt")
pos_texts = []

for filename in pos_files:
    text = open(filename).read()
    pos_texts.append(text)


neg_files = glob.glob("C:/Users/Gabriela/Desktop/classificacao/negativos/*.txt")
neg_texts = []

for filename in neg_files:
    text = open(filename).read()
    neg_texts.append(text)


X = pos_texts + neg_texts
Y = [1]*120 + [0]*120


result_x = np.array(X)
result_y = np.array(Y)

count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(X)

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_count)

classificador = svm.SVC(gamma='auto')

classificador.fit(X_tfidf, result_y)

example = ["Apenas um favelado mt bem sucedido"]
example_count = count_vectorizer.transform(example)
example_tfidf = tfidf.fit_transform(example_count)
result = classificador.predict(example_tfidf)
print(result)
if result == 0:
    print("Este comentário foi racista")
else:
    print("Este comentário não foi racista")
