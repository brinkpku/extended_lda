# !/usr/bin/env python3
import utils

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV


@utils.timer
def do_lda(input_text, feature_method='tf', topic_num=5, method="batch", max_iter=20):
    '''
    do lda process
    :param input_text: list, [str,], preprocessed text
    :return: tuple of np.array. terms, doc-topic probability, topic-word probability, perplexity
    '''
    x, terms = extract_feature(input_text)
    lda = LatentDirichletAllocation(n_components=topic_num, learning_method=method, max_iter=max_iter, random_state=0,
                                    batch_size=128)
    lda_topics = lda.fit_transform(x)
    return np.array(terms), lda_topics, lda.components_, lda.perplexity(x)

@utils.timer
def extract_feature(input_text, method="tf"):
    """ extract text feature, support tf and tf*idf
    input_text: list of str, docs in list
    method: str, 'tf' or 'idf'
    return: (vector, feature_names)
    """
    vector = None
    if method == 'tf':

        vector = CountVectorizer(ngram_range=(1, 1), stop_words='english')
        vector.build_analyzer()
    elif method == 'idf':
        vector = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        vector.build_analyzer()
    else:
        print("unknown method")
        return
    return vector.fit_transform(input_text), vector.get_feature_names()


def generate_lda_parameter(min_topics, max_topics, step, max_iter=1000):
    """ generate lda parameter for grid search
    min_topics: int
    max_topics: int
    step: int
    max_iter: int, iter time
    """
    return {
        'n_components': range(min_topics, max_topics, step),
        'max_iter': [max_iter],
        "random_state": [0],
        "batch_size": [128],
        "learning_method":["online"],
        'learning_decay': [.5, .7, .9],
    }

@utils.timer
def gridsearchCV(parameters, data):
    """ use GridSearchCV to find best model
    parameters: dict, model parameters
    data: train data
    return: sklearn.model_selection.GridSearchCV, fit model
    """
    lda = LatentDirichletAllocation()
    model = GridSearchCV(lda, parameters, n_jobs=-1) # parellel with multi-processors
    model.fit(data)
    return model


def print_topics(topic_word, terms, doc_topic, num=20):
    '''
    print topics
    :param topic_word: np.array, topic-word probability
    :param terms: np.array, feature names
    :param doc_topic: np.array, doc-topic probability
    :param num: int, term num/doc num of topic to print
    :return: None
    '''
    for idx, t in enumerate(topic_word):
        sort_word_idx = np.argsort(t)
        print("#", idx + 1, "-" * 20)
        for iidx in sort_word_idx[-1:-num - 1:-1]:
            print(":".join([terms[iidx], str(t[iidx])]))
        sort_doc_idx = np.argsort(doc_topic[:, idx])
        for iidx in sort_doc_idx[-1:-num - 1:-1]:
            print(":".join([str(iidx), str(doc_topic[iidx][idx])]))


def get_topics(topic_word, terms, doc_topic, num=20):
    '''
    like print_topics, but return values
    :param topic_word: np.array, topic-word probability
    :param terms: np.array, feature names
    :param doc_topic: np.array, doc-topic probability
    :param num: int, term num/doc num of topic to print
    :return: tuple of list.
    '''
    top_terms = []
    top_docs = []
    for idx, t in enumerate(topic_word):
        sort_word_idx = np.argsort(t)
        top_term = []
        for iidx in sort_word_idx[-1:-num - 1:-1]:
            top_term.append((terms[iidx], t[iidx]))
        top_terms.append(top_term)
        top_doc = []
        sort_doc_idx = np.argsort(doc_topic[:, idx])
        for iidx in sort_doc_idx[-1:-num - 1:-1]:
            top_doc.append((iidx, doc_topic[iidx][idx]))
        top_docs.append(top_doc)
    return top_terms, top_docs
