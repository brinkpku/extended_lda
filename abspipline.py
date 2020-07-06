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
    model_name = "abstractlda614c_v2001"
    raw_data = persister.load_json(configs.RAWABSTRACT)
    lda_input = persister.read_input(configs.ABSTRACTINPUT)
    terms, doc_topic, topic_word = persister.read_lda(configs.ABSTRACTLDA.format(model_name))
    absparse = persister.read_parse(configs.ABSTRACTPARSE)
    top_terms, top_docs = lda.get_topics(topic_word, terms, doc_topic)
    for idx, t in enumerate(top_terms):
        vis.draw_word_cloud(configs.WORDCLOUD.format("abs", idx), {terms[k]:v for k,v in top_terms[idx]})
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
            res = relation.corenlp_annotate(client, pp.format_abs(abstract))
            persister.add_json(configs.ABSTRACTPARSE, res)

elif configs.MODE == "reparse":
    print("get raw data..")
    raw_data = persister.load_json(configs.RAWABSTRACT)
    parseres = persister.read_parse(configs.ABSTRACTPARSE)
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    relation.reannotate(failed_idxs, configs.ABSTRACTPARSE,
                        raw_data, format_func=pp.format_abs)

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
            handled_text = pp.lemma_texts(pp.format_abs(raw_data[idx]))
        else:
            print("convert to lda input:{}/{}".format(idx, len(absparse) - 1))
            handled_text = pp.convert_parse2lda_input(parsed)
        preprocessed = pp.preprocess(handled_text)
        persister.add_input(configs.ABSTRACTINPUT, " ".join(preprocessed))

elif configs.MODE == "tune":
    print("do lda..")
    abs_input = persister.read_input(configs.ABSTRACTINPUT)
    min_df = 1
    measure = "c_uci"
    tf, vec_model = lda.extract_feature(abs_input, min_df=min_df)
    terms = np.array(vec_model.get_feature_names())
    learning_rates = [.6, ]
    topic_params = range(10, 50, 5)
    line_data = []
    best_model = None
    best_rate = .6
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
    model_name = configs.ABSTRACTMODEL.format(
        int(best_rate*10), len(topic_word), measure, best_iter, min_df)
    persister.persist_lda(configs.ABSTRACTLDA.format(
        model_name), terms, doc_topic, topic_word)
    persister.save_model(model_name, best_model)
    persister.save_model(configs.NEWSVEC.format(min_df), vec_model)
    # lda.print_topics(topic_word, terms, doc_topic)
    vis.plot_line(configs.TUNELINE.format("abs", measure), line_data, list(map(str, learning_rates)),
                  xlabel="topic num", ylabel="{} coherence".format(measure), 
                  legend_title="learning decay", title="best model:{}".format(model_name))

elif configs.MODE == "lda":
    _rate = .6
    _iter = 200
    _num = 14
    measure = "c_v"
    min_df = 1
    model_name = configs.ABSTRACTMODEL.format(
        int(_rate*10), _num, measure, _iter, min_df)
    print("train", model_name)
    _input = persister.read_input(configs.ABSTRACTINPUT)
    tf, vec_model = lda.extract_feature(_input, min_df=min_df)
    terms = np.array(vec_model.get_feature_names())
    doc_topic, lda_model = lda.do_lda(
        tf, topic_num=_num, max_iter=_iter, learning_decay=_rate)
    persister.persist_lda(configs.ABSTRACTLDA.format(
        model_name), terms, doc_topic, lda_model.components_)
    persister.save_model(model_name, lda_model)
    persister.save_model(configs.ABSVEC.format(min_df), vec_model)
    print("Done")
