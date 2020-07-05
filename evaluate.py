# !/usr/bin/env python3

""" evaluate expriments results
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC

import lda


def filter_triple2term_idx(lemma_triple, vec):
    """ filter triple words by vocabulary and return its idx in vocabulary
    triple: list of tuple of str, triple of lemma words
    vec: countvector model
    return: list of int, word idx in vocabulary
    """
    flatten_triple = [lemma for argument in lemma_triple for lemma in argument]
    return [int(vec.vocabulary_[lemma]) for lemma in flatten_triple if lemma in vec.vocabulary_]


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


def get_topic_term_frequency(topic_texts, min_df=1):
    vector = CountVectorizer(ngram_range=(
        1, 1), stop_words='english', min_df=min_df)
    vector.build_analyzer()
    tf = vector.fit_transform(topic_texts)
    return tf.toarray().sum(axis=0), vector


def get_topic_term_tfidf(topic_texts, min_df=1):
    vector = TfidfVectorizer(ngram_range=(
        1, 1), stop_words='english', min_df=min_df)
    vector.build_analyzer()
    tfidf = vector.fit_transform(topic_texts)
    return tfidf.toarray().sum(axis=0), vector


def sort_words(words, values, vec):
    """ 对词语按照tf或idf值原地排序
    words: list of str
    values: np.array, tf or idf value
    terms: np.array of str, vocabulary
    return: None, 修改words数组
    """
    words.sort(key=lambda x: values[vec.vocabulary_[x]], reverse=True)


def evaluate_triples_by_coherence(extended_res, vec, terms, topic_word, input_text, top_docs, measure, window_size=None, score="tf"):
    """ 
    extended_res: relation.extend_lda_results return
    vec: countvector model
    return: (raw_score, extend_score)
    """
    tf = vec.fit_transform(input_text)
    fake_idxs = []
    min_top = 9999
    for idx, e in enumerate(extended_res):  # topic
        topic_texts = [input_text[top_doc[0]] for top_doc in top_docs[idx]]
        topic_term_fre, topic_vec_tf = get_topic_term_frequency(topic_texts)
        # tfidf, vec_tfidf = get_topic_term_tfidf(topic_texts)
        new_words = []
        for t in e:  # triples
            new_words.extend([lemma for argument in t[0]
                              for lemma in argument])
        new_words = [word for word in list( # remove duplicate and not in vocab
            set(new_words)) if word in topic_vec_tf.vocabulary_]
        if score == "tf":
            sort_words(new_words, topic_term_fre, topic_vec_tf)
        else:
            raise ValueError("unknown sort method")
        if len(new_words) < min_top:
            min_top = len(new_words)
        fake_idxs.append([vec.vocabulary_[word] for word in new_words])
    fake_topic_word = gen_fake_topic_word(topic_word.shape, fake_idxs)
    return lda.get_coherence(tf, terms, fake_topic_word, input_text, measure, top_n=min_top, window_size=window_size)


def evaluate_triples_by_vec_dist():
    pass


def evaluate_classify():
    pass