##Detecção de insultos racistas no Twitter

import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


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

classifier = MultinomialNB()
classifier.fit(X_tfidf, y)
#xd = classifier.score(X_tfidf, y)

example = ["O homem negro não foi feito para arrastar correntes e sim para voar no meio da sociedade."]
example_count = count_vectorizer.transform(example)
example_tfidf = tfidf.transform(example_count)
result = classifier.predict(example_tfidf)
print(result)
if result == 0:
    print("Este comentário foi racista")
else:
    print("Este comentário não foi racista")
#print("Precisão: %.2f"%xd)

