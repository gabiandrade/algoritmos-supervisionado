## Detecção de insultos racistas no Twitter

import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

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
y = [1]*120 + [0]*120

count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(X)

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_count)

#Cria o objeto de regressão logística
model = LogisticRegression()
model.fit(X_tfidf, y)

#Coeficiente da equação e intercepto
#print('Coefficient: \n', model.coef_)
#print('Intercept: \n', model.intercept_)

example = ["O homem negro não foi feito para arrastar correntes e sim para voar no meio da sociedade."]
example_count = count_vectorizer.transform(example)
example_tfidf = tfidf.transform(example_count)
#xd = model.score(example_tfidf, y)
#Prevê o resultado
predicted= model.predict(example_tfidf)
print(predicted)
if predicted == 0:
    print("Este comentário foi racista")
else:
    print("Este comentário não foi racista")

#print("Precisão: %.2f"%xd)
