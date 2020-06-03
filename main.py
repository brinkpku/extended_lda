# !/usr/bin/env python3

import time

import numpy as np
import pandas as pd
from stanza.server import CoreNLPClient

import preprocess as pp
import lda
import configs
import persister
import relation


if configs.MODE == "load":
    print("load mode..")
    raw_data = persister.load_json(configs.RAWABSTRACT)
    lda_input = persister.read_input(configs.ABSTRACTINPUT)
    terms, doc_topic, topic_word = persister.read_lda(configs.ABSTRACTLDA)
    absparse = persister.read_parse(configs.ABSTRACTPARSE)
elif configs.MODE == "init":
    # DE: keywords
    # ID: extended keywords
    # TI: title
    # AB: abstract
    print('load data..')
    whole_data = None
    for i in range(1, 7):
        df = pd.read_csv('data/{}.txt'.format(str(i)),
                         delimiter='\t',
                         usecols=['DE', 'ID', 'TI', 'AB'],
                         encoding='utf8',
                         index_col=False,
                         dtype=np.str)
        df = df[df['AB'].notnull() & df['TI'].notnull()
                ]  # filter null abstract
        df = df.fillna('')
        if whole_data is None:
            whole_data = df
        else:
            whole_data = pd.concat([whole_data, df], ignore_index=True)
    raw_data = list(whole_data["AB"])
    persister.save_json(configs.RAWABSTRACT, raw_data)

    print("annotate sentence..")
    with CoreNLPClient(properties="./corenlp_server.props", timeout=30000, memory='4G') as client:
        for idx, abstract in enumerate(raw_data):
            if idx < int(configs.RECOVERIDX):
                print("recover", idx)
                continue
            print("parse {}/{} abstract".format(idx, len(raw_data) - 1))
            res = relation.corenlp_annotate(client, abstract)
            persister.add_json(configs.ABSTRACTPARSE, res)
    absparse = persister.read_parse()

    print("preprocess as lda input..")
    for idx, parsed in enumerate(absparse):
        if type(parsed) == str:
            print("{} no parse result, use raw text instead of lemmatized.".format(idx))
            handled_text = " ".join(pp.preprocess_abstract(raw_data[idx]))
        else:
            print("convert to lda input:{}/{}".format(idx, len(absparse) - 1))
            handled_text = " ".join(
                [" ".join([w["lemma"] for w in sent["tokens"]]) for sent in parsed["sentences"]])
        preprocessed = pp.preprocess_abstract(handled_text)
        persister.add_input(configs.ABSTRACTINPUT, " ".join(preprocessed))
    lda_input = persister.read_input(configs.ABSTRACTINPUT)

    print("do lda..")
    # vocab = keywords
    vocab = None
    topic_param = 20
    terms, doc_topic, topic_word, perplexity = lda.do_lda(
        lda_input, 'tf', topic_param, vocab)
    persister.persist_lda(configs.ABSTRACTLDA, terms, doc_topic, topic_word)

lda.print_topics(topic_word, terms, doc_topic)
