import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words

model = Word2Vec.load("data/word_model.mod")

d1 = "Government speaks to the media in Illinois"

# d2 = "The president addresses the press"
# d2 = "The state addresses the press"
d2 = "The state addresses the press in Chicago"


vocabulary = [w for w in set(d1.lower().split() + d2.lower().split()) if w in model.vocab and w not in stop_words.ENGLISH_STOP_WORDS]
vect = CountVectorizer(vocabulary=vocabulary).fit([d1, d2])

from sklearn.metrics import euclidean_distances
W_ = np.array([model[w] for w in vect.get_feature_names() if w in model])
D_ = euclidean_distances(W_)
D_ = D_.astype(np.double)
D_ /= D_.max()  # just for comparison purposes

from scipy.spatial.distance import cosine
v_1, v_2 = vect.transform([d1, d2])
v_1 = v_1.toarray().ravel()
v_2 = v_2.toarray().ravel()
print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))

from pyemd import emd
# pyemd needs double precision input
v_1 = v_1.astype(np.double)
v_2 = v_2.astype(np.double)
v_1 /= v_1.sum()
v_2 /= v_2.sum()
print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))
