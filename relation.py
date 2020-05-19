# !/usr/bin/env python3

"""
find semantic relation from documents
"""

import preprocess as pp

from stanfordcorenlp import StanfordCoreNLP

# CLI = StanfordCoreNLP('http://localhost', port=9000)
CLI = StanfordCoreNLP(r'/mnt/d/stanford-corenlp-full-2018-02-27', port=9001)
print("inited stanford CoreNLP client, dont forget to close it!")

# clean up
def close_cli():
    CLI.close()
    print("closed stanford CoreNLP client!")
close_cli()
# extract sentence
def lemmatize_sent_words(sent):
    """ word_tokenize sentences(use corenlp) and lemmatize(use nltk)
    return: list of str
    """
    return [pp.lemmatize(word) for word in CLI.word_tokenize(sent)]

def get_sent_idx(lemmatized_sents_words, topic_word):
    """ get idx of sentence contains topic word
    lemmatized_sents_words: list, [[lemmatized_word,..],[...]]
    return: set of int, indexs
    """
    idxs = set()
    for idx, sent in enumerate(lemmatized_sents_words):
        if topic_word in sent:
            idxs.add(idx)
    return idxs

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
    



if __name__ == '__main__':
    sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    print('Tokenize:', CLI.word_tokenize(sentence))
    print('Part of Speech:', CLI.pos_tag(sentence))
    print('Named Entities:', CLI.ner(sentence))
    print('Constituency Parsing:', CLI.parse(sentence))
    print('Dependency Parsing:', CLI.dependency_parse(sentence))
    close_cli()
