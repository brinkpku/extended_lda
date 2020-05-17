#!/usr/bin/env python3

import re
from string import punctuation


from nltk import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import utils


WNL = WordNetLemmatizer()
URL_REG = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'


def lemmatize(word):
    return WNL.lemmatize(word)


def preprocess_abstract(abstract):
    '''
    preprocess abstract: lower, remove punctuations, tokenize
    :param abstract: str
    :return: list
    '''
    abstract = re.sub(URL_REG, ' ', abstract)
    abstract = re.sub('\d+?', ' ', abstract)
    for p in punctuation:
        abstract = re.sub(re.escape(p), ' ', abstract)
    abstract = abstract.lower()
    abstract = [lemmatize(w) for w in word_tokenize(abstract)]
    filtered = [w for w in abstract if w not in stopwords.words('english')]
    return filtered


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
