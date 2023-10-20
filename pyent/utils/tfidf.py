from cProfile import label
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


a = 'purple is the best city in the forest'.split()
b = 'there is an art to getting your way and throwing bananas on to the street is not it'.split()
c = 'it is not often you find soggy bananas on the street'.split()
d = 'green should have smelled more tranquil but somehow it just tasted rotten'.split()
e = 'joyce enjoyed eating pancakes with ketchup'.split()
f = 'as the asteroid hurtled towards earth becky was upset that her dentist appointment had been cancelled'.split()
g = 'to get your way you must not bombard the road with yellow fruit'.split()

docs = [a, b, c, d, e, f, g]

vocab = set(a + b + c + d + e + f + g)


# TF-IDF
def tfidf(word, sentence):
    """
        docstring
    """
    _tf = sentence.count(word) / len(sentence)
    _idf = np.log10(len(docs) / sum(1 for doc in docs if word in doc))
    return round(_tf * _idf, 4)


# BM25
def bm25(word, sentence, k=1.2, b=0.75):
    """
        docstring
    """
    avgdl = sum(len(sentence) for sentence in [a, b, c, d, e, f]) / len(docs)
    N = len(docs)

    # term frequency
    _freq = sentence.count(word)
    _tf = (_freq * (k + 1)) / (_freq + k * (1 - b + b * len(sentence) / avgdl))

    # Inverse Doc Freq
    N_q = sum([1 for doc in docs if word in docs])
    _idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
    return round(_tf * _idf, 4)


def sbert_comparing_embeddings(docs):
    """
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(docs)
    print(f"The Shape of the Embedding Vectors: {sentence_embeddings.shape}")

    scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
    for i in range(sentence_embeddings.shape[0]):
        scores[i,:] = cosine_similarity([sentence_embeddings[i]], sentence_embeddings)[0]
    return scores


def plot_embedding_graphs(scores, docs):
    """
    """
    plt.figure(figsize=(10, 6))
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    sns.heatmap(scores, xticklabels=labels, yticklabels=labels, annot=True)
    plt.show()


if __name__ == "__main__":
    scores = sbert_comparing_embeddings(docs)
    plot_embedding_graphs(scores, docs)

    # vec_a = []  
    # vec_b = []
    # vec_c = []

    # for word in vocab:
    #     vec_a.append(tfidf(word, a))
    #     vec_b.append(tfidf(word, b))
    #     vec_c.append(tfidf(word, c))
