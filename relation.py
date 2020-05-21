# !/usr/bin/env python3

"""
find semantic relation from documents
"""
import configs
import preprocess as pp

import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from sklearn.preprocessing import MinMaxScaler

# corenlp api
CLI = StanfordCoreNLP('http://localhost', port=9001)
# CLI = StanfordCoreNLP(r'/mnt/d/stanford-corenlp-full-2018-02-27', port=9001)
print("inited stanford CoreNLP client, dont forget to close it!")
# clean up


def close_cli():
    CLI.close()
    print("closed stanford CoreNLP client!")


# not use cli, just close it
if not configs.USECLI:
    close_cli()


def corenlp_annotate(cli, text):
    """ use corenlp annotate(self defined pipeline) sentence
    cli: stanza corenlp client
    text: str, sentences
    return: str, json res if succeeded else error msg
    """
    try:
        res = cli.annotate(text)
        return res.json()
    except Exception as err:
        return "err: {}".format(err)


# relation extraction
# extract sentence
def lemmatize_sent_words(sent):
    """ word_tokenize sentences(use corenlp) and lemmatize(use nltk)
    return: list of str
    """
    return [pp.lemmatize(word) for word in CLI.word_tokenize(sent)]


def get_sents_idx(lemmatized_sents_words, topic_word):
    """ get idxs of sentences contain topic wor
    use extract_important_sents to get more meaning for result
    lemmatized_sents_words: list, [[lemmatized_word,..],[...]]
    return: set of int, indexs
    """
    idxs = set()
    for idx, sent in enumerate(lemmatized_sents_words):
        if topic_word in sent:
            idxs.add(idx)
    return idxs


def get_topic_word_idx(sent, topic_words):
    """ get idxs of topic words which are contained in the sent
    idx is index of word in topic_words
    sent: list of str, [lemmatized_word,..]
    topic_words: list of str
    return: set of int, indexs
    """
    idxs = set()
    for idx, w in enumerate(topic_words):
        if w in sent:
            idxs.add(idx)
    return idxs


SCALER = MinMaxScaler()


def min_max(raw_data):
    """https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn-preprocessing-minmaxscaler
    raw_data: {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
    """
    return SCALER.fit_transform(raw_data)


def extract_important_sents(sents, topic_words, components_values):
    """ extract important sentences idx from a news/abstract by evaluating topic word
    from less important to more important
    sentences: list, [[lemmatized_word,..],[...]]
    topic_words: list of str
    components_values: np.array, [can be viewed as pseudocount that represents the number of times word
                                was assigned to topic ], normalized
    return: tuple. (np.array, importance, count), important sentences idx, importance, count
    """
    count = []
    importance = []
    normalized_components_values = min_max([[c] for c in components_values])
    for sent in sents:
        contained_word_idxs = get_topic_word_idx(sent, topic_words)
        # sum normalized components value as importance
        importance.append(sum([normalized_components_values[i][0] for i in contained_word_idxs]))
        count.append(len(contained_word_idxs))
    return np.argsort(importance)[-1::-1], importance, count


# extract word relation
def extract_word_relation_from_sent(topic_word_idx, parse_res):
    """ get all dependency relation of topic word from one sentence
    topic_word_idx: int, index of topic word in sentence
    parse_res: list of dict, [sent1{dep parse res}]
    return: list
    """
    topic_word_idx += 1 # root node is 0
    relations = []
    for r in parse_res:
        if topic_word_idx == r["dependent"]:
            relations.append(r)
        elif topic_word_idx == r["governor"]:
            relations.append(r)
    return relations




if __name__ == '__main__':
    sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    print('Tokenize:', CLI.word_tokenize(sentence))
    print('Part of Speech:', CLI.pos_tag(sentence))
    print('Named Entities:', CLI.ner(sentence))
    print('Constituency Parsing:', CLI.parse(sentence))
    print('Dependency Parsing:', CLI.dependency_parse(sentence))
    close_cli()
