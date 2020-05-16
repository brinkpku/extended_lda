# !/usr/bin/env python3

import time

import numpy as np
import pandas as pd

import preprocess as pp
import lda
import configs
import persister


if configs.MODE == "load":
    print("load mode..")
    input_text = persister.load_json(configs.ABSTRACTDATA)
    abssent = persister.load_json(configs.ABSTRACTSENT)
    terms, doc_topic, topic_word = persister.read_lda(configs.ABSTRACTLDA)
# DE: keywords
# ID: extended keywords
# TI: title
# AB: abstract
elif configs.MODE == "init":
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
    # whole_data.info()
    # whole_data.head()
    # print("get keywords as vocab..")
    # keywords = pp.preprocess_keywords(whole_data["DE"])
    # save sentences 
    abssent = []
    for abs in whole_data:
        abssent.append(pp.split2sent(abs))
    persister.save_json(configs.ABSTRACTSENT, abssent)

    print("preprocess data..")
    input_text = [' '.join(pp.preprocess_abstract(a))
                  for a in whole_data['AB']]
    persister.save_json(configs.ABSTRACTDATA, input_text)
    
    print("do lda..")
    # vocab = keywords
    vocab = None
    topic_param = 20
    terms, doc_topic, topic_word, perplexity = lda.do_lda(
        input_text, 'tf', topic_param, vocab)
    persister.persist_lda(configs.ABSTRACTLDA, terms, doc_topic, topic_word)

lda.print_topics(topic_word, terms, doc_topic)
