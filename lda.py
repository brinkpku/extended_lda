# !/usr/bin/env python3
import utils

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel


################################
# model and parameters tunning #
################################
@utils.timer
def gensim_lda(texts):
    """ use gensim train lda model, not finished
    texts: list of list of str, tokenized text
    """
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
    return model


@utils.timer
def do_lda(input_vector, feature_method='tf', topic_num=5, method="online", max_iter=20, learning_decay=0.7):
    '''
    do lda process
    :param input_vector: list, sklearn CountVectorizer output
    :return: tuple of np.array. doc-topic probability, sklearn.decomposition.LatentDirichletAllocation
        topic-word probability and perplexity can be got from model by model.n_components
    '''
    model = LatentDirichletAllocation(n_components=topic_num, learning_method=method, max_iter=max_iter, random_state=0,
                                    batch_size=128, learning_decay=learning_decay)
    lda_topics = model.fit_transform(input_vector)
    return lda_topics, model


@utils.timer
def select_model_by_perplexity(tf, topic_num, max_iter=20, learning_decay=0.7):
    """ compute different model perlexity and save best model
    tf: list, count-vector
    topic_num: iterator, different topic num
    return: best model, perplexity list
    """
    perplexity_lst = []
    best_model = None
    for num in topic_num:
        _, model = do_lda(tf, topic_num=num, max_iter=max_iter, learning_decay=learning_decay)
        perp = model.perplexity(tf)
        if not perplexity_lst or perp < min(perplexity_lst):
            best_model = model
        print("\ttest {} topics, perplexity:{}".format(num, round(perp, 2)))
        perplexity_lst.append(perp)
    return best_model, perplexity_lst


@utils.timer
def extract_feature(input_text, method="tf", min_df=1):
    """ extract text feature, support tf and tf*idf
    input_text: list of str, docs in list
    method: str, 'tf' or 'idf'
    min_df: int or float, refer to sklearn.CounVectorizer
    return: (vector, feature_names)
    """
    vector = None
    if method == 'tf':
        vector = CountVectorizer(ngram_range=(1, 1), stop_words='english', min_df=min_df)
        vector.build_analyzer()
    elif method == 'idf':
        vector = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', min_df=min_df)
        vector.build_analyzer()
    else:
        print("unknown method")
        return
    return vector.fit_transform(input_text), np.array(vector.get_feature_names())


def generate_lda_parameter(min_topics, max_topics, step, max_iter=[1000]):
    """ generate lda parameter for grid search
    min_topics: int
    max_topics: int
    step: int
    max_iter: int, iter time
    """
    return {
        'n_components': range(min_topics, max_topics, step),
        'max_iter': max_iter,
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


@utils.timer
def select_model_by_coherence(tf, terms, texts, measure, topic_num, top_n=20, max_iter=20, learning_decay=0.7):
    """ use tmtool to compute different model coherence and save best model
    tf: list, count-vector
    terms: np.array of str
    texts: list of list of str, tokenized text
    measure: str, c_v, u_mass, c_uci, c_npmi
    topic_num: iterator, different topic num
    return: best model, coherence list
    """
    c_lst = []
    best_model = None
    for num in topic_num:
        _, model = do_lda(tf, topic_num=num, max_iter=max_iter, learning_decay=learning_decay)
        coherences = get_coherence(tf, terms, model.n_components, texts, measure)
        c = np.mean(coherences)
        if not c_lst or c > max(c_lst):
            best_model = model
        print("\ttest {} topics, each coherence:{}, avg coherence:{}".format(num, coherences, round(c, 2)))
        c_lst.append(c)
    return best_model, c_lst
    

def get_coherence(tf, terms, topic_word, texts, measure, top_n=20, window_size=None):
    """ use tmtool get coherence
    tf: list, count-vector
    terms: np.array of str
    texts: list of list of str, tokenized text
    measure: str, c_v, u_mass, c_uci, c_npmi
    top_n: int, top words selected from topic
    window_size: int, c_something method used for compute probability
    return coherence list
    """
    return metric_coherence_gensim(measure=measure, 
                        top_n=top_n, 
                        topic_word_distrib=topic_word, 
                        dtm=tf, 
                        vocab=terms, 
                        texts=texts)



##############################################
# lda result process, analysis, visulization #
##############################################
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
