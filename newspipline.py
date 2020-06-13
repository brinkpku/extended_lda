# !/usr/bin/env python3
import json
import time
import os

from stanza.server import CoreNLPClient
import numpy as np

from newsgroups import get_news_data, CATEGORIES
import preprocess as pp
import lda
import configs
import persister
import relation
import vis

if configs.MODE == "load":
    print("load mode..")
    rawnews = persister.load_json(configs.RAWNEWS)
    lda_input = persister.read_input(configs.NEWSINPUT)
    terms, doc_topic, topic_word = persister.read_lda(configs.NEWSLDA)
    newsparse = persister.read_parse()
elif configs.MODE == "init":  # prepare and preprocess raw data
    print("get raw data..")
    raw_data = get_news_data(500)
    rawnews = []
    for cate in raw_data:
        rawnews.extend(cate)
    persister.save_json(configs.RAWNEWS, rawnews)

elif configs.MODE == "parse":  # parse raw data with corenlp annotator
    print("annotate sentence..")
    rawnews = persister.load_json(configs.RAWNEWS)
    with CoreNLPClient(properties="./corenlp_server.props", timeout=25000, memory='4G') as client:
        for idx, news in enumerate(rawnews):
            if idx < int(configs.RECOVERIDX):
                print("recover", idx)
                continue
            print("parse {}/{} news".format(idx, len(rawnews) - 1))
            res = relation.corenlp_annotate(client, pp.format_news(news))
            persister.add_json(configs.NEWSPARSE, res)

elif configs.MODE == "reparse":  # reparse failed data
    print("get raw data..")
    rawnews = persister.load_json(configs.RAWNEWS)
    parseres = persister.read_parse()
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    time.sleep(10)
    relation.reannotate(failed_idxs, configs.NEWSPARSE, rawnews)

elif configs.MODE == "preprocess":  # use parsed lemma and preprocess it as lda input
    print("preprocess as lda input..")
    if os.path.exists(configs.NEWSINPUT):
        os.remove(configs.NEWSINPUT)
        print("removed", configs.NEWSINPUT)
    newsparse = persister.read_parse()
    rawnews = persister.load_json(configs.RAWNEWS)
    for idx, parsed in enumerate(newsparse):
        if type(parsed) == str:
            print("{} no parse result, use nltk lemmatized text instead.".format(idx))
            tmp = pp.format_news(rawnews[idx])
            handled_text = pp.lemma_texts(tmp)
        else:
            print("convert to lda input:{}/{}".format(idx, len(newsparse) - 1))
            handled_text = pp.convert_parse2lda_input(parsed)
        preprocessed = pp.preprocess(handled_text)
        persister.add_input(configs.NEWSINPUT, " ".join(preprocessed))

elif configs.MODE == "tune":  # lda model tuning
    print("run lda..")
    news_input = persister.read_input(configs.NEWSINPUT)
    min_df = 1
    measure = "c_v"
    tf, vec_model = lda.extract_feature(news_input, min_df=min_df)
    terms = np.array(vec_model.get_feature_names())
    max_iter_times = [200, 500]
    learning_rates = [.5, .6, .7, .8, .9]
    best_model = None
    best_res = None
    best_rate = .5
    best_iter = 200
    max_c = -999
    line_data = []
    for itertime in max_iter_times:
        c_lst = []
        for r in learning_rates:
            doc_topic, lda_model = lda.do_lda(tf, topic_num=len(
                CATEGORIES), max_iter=itertime, learning_decay=r)
            coherences = lda.get_coherence(
                tf, terms, lda_model.components_, news_input, measure)
            c = np.mean(coherences)
            if c > max_c:
                best_model = lda_model
                best_res = doc_topic
                best_rate = r
                best_iter = itertime
                max_c = c
            print("\ttest {} iter, each coherence:{}, avg coherence:{}".format(
                itertime, coherences, round(c, 2)))
            c_lst.append(c)
        line_data.append(list(zip(learning_rates, c_lst)))
    topic_word = best_model.components_
    persister.persist_lda(configs.NEWSLDA, terms, best_res, topic_word)
    model_name = configs.NEWSMODEL.format(int(best_rate*10), len(topic_word), measure, best_iter)
    persister.save_model(model_name, best_model)
    persister.save_model(configs.NEWSVEC.format(min_df), vec_model)
    lda.print_topics(topic_word, terms, best_res)
    vis.plot_line("news_iter_learning", line_data, list(map(str, max_iter_times)),
                  xlabel="learning decay", ylabel="coherence", legend_title="iter time", title=model_name)

else:
    print("unknown mode.")
