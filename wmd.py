import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.metrics import euclidean_distances
from pyemd import emd

model = Word2Vec.load("data/word_model.mod")

def get_wmd_distance(d1, d2, min_vocab=7, verbose=False):
	vocabulary = [w for w in set(d1.lower().split() + d2.lower().split()) if w in model.vocab and w not in stop_words.ENGLISH_STOP_WORDS]
	if len(vocabulary) < min_vocab:
		return 1
	vect = CountVectorizer(vocabulary=vocabulary).fit([d1, d2])
	W_ = np.array([model[w] for w in vect.get_feature_names() if w in model])
	D_ = euclidean_distances(W_)
	D_ = D_.astype(np.double)
	D_ /= D_.max()  # just for comparison purposes
	v_1, v_2 = vect.transform([d1, d2])
	v_1 = v_1.toarray().ravel()
	v_2 = v_2.toarray().ravel()
	# pyemd needs double precision input
	v_1 = v_1.astype(np.double)
	v_2 = v_2.astype(np.double)
	v_1 /= v_1.sum()
	v_2 /= v_2.sum()
	if verbose:
		print vocabulary
		print v_1, v_2
	return emd(v_1, v_2, D_)

# d1 = "Government speaks to the media in Illinois"
# d2 = "The state addresses the press in Chicago"
# print get_wmd_distance(d1, d2)
