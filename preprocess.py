#!/usr/bin/env python3

import re
from string import punctuation


from nltk import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

import utils


WNL = WordNetLemmatizer()
URL_REG = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
EMAIL_REG = "[0-9a-zA-Z_]{0,19}@(?:[0-9a-zA-Z]{1,13}[.])+\w+"


def lemmatize(word, pos=None):
    if pos:
        return WNL.lemmatize(word, pos)
    else:
        return WNL.lemmatize(word)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_abstract(abstract):
    '''
    preprocess abstract as lda input: lower, remove punctuations, tokenize
    :param abstract: str
    :return: list
    '''
    abstract = re.sub(URL_REG, ' ', abstract)
    abstract = re.sub(EMAIL_REG, " ", abstract)
    abstract = re.sub('\d+?', ' ', abstract)
    for p in punctuation:
        abstract = re.sub(re.escape(p), ' ', abstract)
    abstract = abstract.lower()
    abstract = [lemmatize(w) for w in word_tokenize(abstract)]
    filtered = [w for w in abstract if w not in stopwords.words('english')]
    return filtered

def format_news(news):
    """ format news, remove unused character
    """
    news = re.sub("<", " ", news)
    news = re.sub(r"\s+", " ", news)
    return news


@utils.timer
def preprocess_keywords(raw_list):
    tmp_set = set()
    for r in raw_list:
        for keyword in r.split(";"):
            processed = " ".join([lemmatize(w)
                                  for w in word_tokenize(keyword.lower())])
            tmp_set.add(processed)
    return sorted(list(tmp_set))


def split2sent(abstract):
    """ split texts to sentences
    return: list of str
    """
    abstract = re.sub(r'\s+', ' ', abstract)
    return sent_tokenize(abstract)


if __name__ == "__main__":
    pass
