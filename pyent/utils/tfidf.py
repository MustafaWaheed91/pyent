from sys import set_coroutine_origin_tracking_depth
import numpy as np
from torch import ResolutionCallback

a = 'purple is the best city in the forest'.split()
b = 'there is an art to getting your way and throwing bananas on to the street is not it'.split()
c = 'it is not often you find soggy bananas on the street'.split()
d = 'green should have smelled more tranquil but somehow it just tasted rotten'.split()
e = 'joyce enjoyed eating pancakes with ketchup'.split()
f = 'as the asteroid hurtled towards earth becky was upset that her dentist appointment had been cancelled'.split()

docs = [a, b, c, d, e, f]


# TF-IDF
def tfidf(word, sentence):
    """
    docstring
    """
    _tf = sentence.count(word) / len(sentence)
    _idf = np.log10(len(docs) / sum(1 for doc in docs if word in doc))
    return round(_tf * _idf, 4)


vocab = set(a + b + c)
vec_a = []  
vec_b = []
vec_c = []

for word in vocab:
    vec_a.append(tfidf(word, a))
    vec_b.append(tfidf(word, b))
    vec_c.append(tfidf(word, c))

# BM25
avgdl = sum(len(sentence) for sentence in [a, b, c, d, e, f]) / len(docs)
N = len(docs)


def bm25(word, sentence, k=1.2, b=0.75):
    # term frequency
    _freq = sentence.count(word)
    _tf = (_freq * (k + 1)) / (_freq + k * (1 - b + b * len(sentence) / avgdl))
    
    # Inverse Doc Freq
    N_q = sum([1 for doc in docs if word in docs])
    _idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
    return round(_tf * _idf, 4)


if __name__ == "__main__":
    print(vec_a)
