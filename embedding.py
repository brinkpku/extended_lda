# !/usr/bin/env python3

"""
embedding methods
"""
import utils
import configs
import persister

import multiprocessing

from gensim.models import word2vec
from nltk import word_tokenize
import numpy as np

@utils.timer
def train_wv(sentences, size=100, window=5, min_count=5):
    '''
    sentences: iterable of iterables, list of tokens
    size: vector length
    window: Maximum distance between the current and predicted word within a sentence
    min_count: Ignores all words with total frequency lower than this.
    return: w2vModel
    '''
    # sg is 0 use cbow model
    w2vModel = word2vec.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
    return w2vModel


def get_doc_dense_vec(model, doc, size=100):
    """ mean vector
    doc: str, lemma text
    return: np.array of np.float32, doc mean vector
    """
    total_words = 0
    s = np.full((1,size), 0, dtype=np.float32)[0]
    for word in word_tokenize(doc):
        if word in model.wv:
            s += model.wv[word]
            total_words += 1
    if total_words == 0:
        return False
    return s / total_words

 
if __name__=="__main__":
    size = 100
    abs_input = persister.read_input(configs.ABSTRACTINPUT)
    # news_input = persister.read_input(configs.NEWSINPUT)

    abs_wv = train_wv([word_tokenize(a) for a in abs_input], size=size)
    abs_wv.save(configs.ABSWV.format(size))

    # news_wv = train_wv([word_tokenize(n) for n in news_input], size=size)
    # news_wv.save(configs.NEWSWV.format(size))
