# !/usr/bin/env python3

"""
find semantic relation from documents
"""
import utils
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
    tmp = 0
    errstr = ""
    while tmp < configs.MAX_TRY:
        tmp += 1
        try:
            res = cli.annotate(text)
            return res.json()
        except Exception as err:
            print("try {}, err: {}, {}.".format(tmp, err, len(text)))
            errstr = "err: {}, {}.".format(err, len(text))
    return errstr
    


# relation extraction
# extract sentence
def lemmatize_sent_words(sent):
    """ word_tokenize sentences(use corenlp) and lemmatize(use nltk)
    return: list of str
    """
    return [pp.lemmatize(word) for word in CLI.word_tokenize(sent)]


def word_in_sents_idx(lemmatized_sents_words, topic_word):
    """ get idxs of sentences contain topic word
    从文本中获取包含“某主题词”的句子的索引
    use extract_important_sents to get more meaning for result
    lemmatized_sents_words: list, [[lemmatized_word,..],[...]]
    return: set of int, indexs
    """
    idxs = set()
    for idx, sent in enumerate(lemmatized_sents_words):
        if topic_word in sent:
            idxs.add(idx)
    return idxs


def get_word_idx(topic_word, sent):
    """ 获取词语在句子中的索引位置
    topic_word: str
    sent: list, tokenized sent
    return: list of int, word index
    """
    res = []
    for idx, i in enumerate(sent):
        if i == topic_word:
            res.append(idx)
    return res



def sent_contained_word_idxs(sent, topic_words):
    """ get idxs of topic words which are contained in the sent
    idx is the index of word in topic_words
    获取句子中所包含的主题词的索引
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
        contained_word_idxs = sent_contained_word_idxs(sent, topic_words)
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
        if topic_word_idx == r[utils.DEPENDENT]:
            relations.append(r)
        elif topic_word_idx == r[utils.GOVERNOR]:
            relations.append(r)
    return relations

def convert_relation2str(relation_dict):
    """ convert dep parse result to string
    relation_dict: dict, dep parse result, 
    return: str
    """
    return utils.DEP2STR.format(relation_dict[utils.GOVERNORGLOSS], relation_dict[utils.DEP], relation_dict[utils.DEPENDENTGLOSS])



if __name__ == '__main__':
    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    # print('Tokenize:', CLI.word_tokenize(sentence))
    # print('Part of Speech:', CLI.pos_tag(sentence))
    # print('Named Entities:', CLI.ner(sentence))
    # print('Constituency Parsing:', CLI.parse(sentence))
    # print('Dependency Parsing:', CLI.dependency_parse(sentence))
    # close_cli()
    import newsgroups
    from stanza.server import CoreNLPClient
    news = newsgroups.get_news_data()
    with CoreNLPClient(properties="./corenlp_server.props", timeout=30000, memory='5G') as client:
        n = pp.format_news(news[0][0])
        print(corenlp_annotate(client, n))
