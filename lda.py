# !/usr/bin/env python3
import utils

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@utils.timer
def do_lda(input_text, feature_method='tf', topic_num=5, vocabulary=None, method="batch"):
    '''
    do lda process
    :param input_text: list, [str,], preprocessed text
    :return: tuple of np.array, terms, doc-topic probability, topic-word probability, perplexity
    '''
    vector = None
    if feature_method == 'tf':
        vector = CountVectorizer(ngram_range=(
            1, 1), vocabulary=vocabulary, stop_words='english')
        vector.build_analyzer()
    if feature_method == 'idf':
        vector = TfidfVectorizer(ngram_range=(
            2, 2), vocabulary=vocabulary, stop_words='english')
        vector.build_analyzer()
    x = vector.fit_transform(input_text)
    lda = LatentDirichletAllocation(n_components=topic_num, learning_method=method, max_iter=20, random_state=0,
                                    batch_size=128)
    lda_topics = lda.fit_transform(x)
    return np.array(vector.get_feature_names()), lda_topics, lda.components_, lda.perplexity(x)


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
        print(terms[sort_word_idx[-1:-num - 1:-1]])
        sort_doc_idx = np.argsort(doc_topic[:, idx])
        print(sort_doc_idx[-1:-num - 1:-1])
