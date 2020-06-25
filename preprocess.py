#!/usr/bin/env python3

import re
from string import punctuation


from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

import utils


WNL = WordNetLemmatizer()
URL_REG = r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
EMAIL_REG = r"[0-9a-zA-Z_.]{0,19}@(?:[0-9a-zA-Z]{1,13}[.])+\w+"
Elsevier = r"\([Cc]\) 20\d\d.+?Elsevier (?:B\.V|Inc|Ltd)\."
Rights = r"All rights reserved\."

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


def lemma_texts(raw_texts):
    """ use nltk lemmatization to handle raw texts
    raw_texts: str, need lightly preprocess, eg. remove empty line (like format_news)
    return: str, lemmatized text
    """
    tokens = pos_tag(word_tokenize(raw_texts))
    res = [lemmatize(pos_res[0], get_wordnet_pos(pos_res[1])) for pos_res in tokens]
    return " ".join(res)



def preprocess(abstract):
    '''
    preprocess abstract as lda input: lower, remove punctuations, stopwords, url, email, tokenize
    remeber to lemmatize text before preprocess.
    :param abstract: str
    :return: list
    '''
    abstract = re.sub(URL_REG, ' ', abstract)
    abstract = re.sub(EMAIL_REG, " ", abstract)
    abstract = re.sub(r'\d+?', ' ', abstract)
    for p in punctuation:
        abstract = re.sub(re.escape(p), ' ', abstract)
    abstract = abstract.lower()
    # abstract = [lemmatize(w) for w in word_tokenize(abstract)]
    abstract = [w for w in word_tokenize(abstract)] # lemmatized in corenlp parse
    filtered = [w for w in abstract if w not in stopwords.words('english')]
    return filtered


def format_abs(abstract):
    """ format abstract, remove corps info
    """
    abstract = re.sub(Elsevier, " ", abstract)
    abstract = re.sub(Rights, " ", abstract)
    abstract = re.sub(r"\s+", " ", abstract)
    return abstract
    

def format_news(news):
    """ format news, remove unused character
    """
    news = re.sub("[<>]|=+", " ", news)
    news = re.sub("-{2,}|/{2,}", " ", news)
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


def convert_parse2lda_input(parsed):
    """ use corenlp lemma result as lda input, need to preprocess
    parsed: dict, stanford corenlp annotated result, use its "sentences" value.
    return: str, lemmatized text
    """
    res = []
    for sent in parsed["sentences"]:
        for w in sent["tokens"]:
            if w["originalText"] in punctuation: # avoid corenlp trans "(" to "-lrb-"
                res.append(w["originalText"])
            else:
                res.append(w["lemma"])
    return " ".join(res)

if __name__ == "__main__":
    pass
