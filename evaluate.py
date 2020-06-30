# !/usr/bin/env python3

""" evaluate expriments results
"""
import numpy as np


def filter_triple2term_idx(lemma_triple, vec):
    """ filter triple words by vocabulary and return its idx in vocabulary
    triple: list of tuple of str, triple of lemma words
    vec: countvector model
    return: list of int, word idx in vocabulary
    """
    flatten_triple = [lemma for argument in lemma_triple for lemma in argument]
    return [int(vec.vocabulary_[lemma]) for lemma in flatten_triple if lemma in vec.vocabulary_ ]


def gen_fake_topic_word(topic_word_shape, fake_idxs):
    """ generate fake topic words distribution by fake word idxs of each topic
    topic_word_shape: tuple, shape of topic_word distribution, k*n, k is topic num, n is word num
    fake_idxs: list of list of int, top_n new word idx of each topic
    return: np.array, shape(k, n, dtype=np.int32), fake topic word distribution 
    """ 
    if topic_word_shape[0] != len(fake_idxs):
        raise ValueError("topic num isn't equal")
    fake_topic_word = np.full(topic_word_shape, -1, dtype=np.int32)
    for tidx in range(topic_word_shape[0]):
        tmp = len(fake_idxs[0])
        for widx in fake_idxs[tidx]:
            fake_topic_word[tidx][widx] = tmp
            tmp -= 1
    return fake_topic_word 