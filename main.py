# !/usr/bin/env python3

import time
import os

import numpy as np
import pandas as pd
from stanza.server import CoreNLPClient

import preprocess as pp
import lda
import configs
import persister
import relation
import vis


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

elif configs.MODE == "parse":
    print("annotate sentence..")
    raw_data = persister.load_json(configs.RAWABSTRACT)
    with CoreNLPClient(properties="./corenlp_server.props", timeout=15000, memory='4G') as client:
        for idx, abstract in enumerate(raw_data):
            if idx < int(configs.RECOVERIDX):
                print("recover", idx)
                continue
            print("parse {}/{} abstract".format(idx, len(raw_data) - 1))
            res = relation.corenlp_annotate(client, abstract)
            persister.add_json(configs.ABSTRACTPARSE, res)

elif configs.MODE == "reparse":
    print("get raw data..")
    raw_data = persister.load_json(configs.RAWABSTRACT)
    parseres = persister.read_parse(configs.ABSTRACTPARSE)
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    relation.reannotate(failed_idxs, configs.ABSTRACTPARSE, raw_data)

elif configs.MODE == "preprocess":
    print("preprocess as lda input..")
    if os.path.exists(configs.ABSTRACTINPUT):
        os.remove(configs.ABSTRACTINPUT)
        print("removed", configs.ABSTRACTINPUT)
    absparse = persister.read_parse(configs.ABSTRACTPARSE)
    raw_data = persister.load_json(configs.RAWABSTRACT)
    for idx, parsed in enumerate(absparse):
        if type(parsed) == str:
            print("{} no parse result, use raw text instead of lemmatized.".format(idx))
            handled_text = pp.lemma_texts(raw_data[idx])
        else:
            print("convert to lda input:{}/{}".format(idx, len(absparse) - 1))
            handled_text = pp.convert_parse2lda_input(parsed)
        preprocessed = pp.preprocess(handled_text)
        persister.add_input(configs.ABSTRACTINPUT, " ".join(preprocessed))

elif configs.MODE == "tune":
    print("do lda..")
    abs_input = persister.read_input(configs.ABSTRACTINPUT)
    min_df = 3
    measure = "c_v"
    tf, vec_model = lda.extract_feature(abs_input, min_df=min_df)
    terms = np.array(vec_model.get_feature_names())
    learning_rates = [.5, .6, .7]
    topic_params = range(8, 22, 2)
    line_data = []
    best_model = None
    best_rate = .5
    best_iter = 200
    c_max = -999
    for r in learning_rates:
        selected_model, c_lst = lda.select_model_by_coherence(
        tf, terms, abs_input, measure, topic_params, learning_decay=r)
        line_data.append(list(zip(topic_params, c_lst)))
        if max(c_lst) > c_max:
            c_max = max(c_lst)
            best_model = selected_model
            best_rate = r
    topic_word = best_model.components_
    doc_topic = best_model.fit_transform(tf)
    persister.persist_lda(configs.ABSTRACTLDA, terms, doc_topic, topic_word)
    model_name = configs.ABSTRACTMODEL.format(int(best_rate*10), len(topic_word), measure, best_iter)
    persister.save_model(model_name, best_model)
    persister.save_model(configs.NEWSVEC.format(min_df), vec_model)
    lda.print_topics(topic_word, terms, doc_topic)
    vis.plot_line("absnums", line_data, list(map(str, learning_rates)),
                  xlabel="topic num", ylabel="coherence", legend_title="learning decay", title=model_name)
