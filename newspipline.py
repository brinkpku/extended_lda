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
    terms, doc_topic, topic_word = persister.read_lda(configs.NEWSLDA.format("newslda74c_v2001"))
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
    # rawnews = persister.load_json(configs.RAWNEWS)
    print("get raw data..")
    rawnews = get_news_data(1000)
    parsed_num = 0
    if os.path.exists(configs.NEWSPARSE+".json"):
        print("load newsparse")
        parseres = persister.read_parse()
        parsed_num = len(parseres)
    real_raw_news = []
    with CoreNLPClient(properties="./corenlp_server.props", timeout=25000, memory='4G') as client:
        while parsed_num<2000:
            cate = rawnews[parsed_num//500]
            for nidx, news in enumerate(cate):
                if nidx<int(configs.RECOVERIDX):
                    print("recover", parsed_num//500, nidx)
                    continue
                print("parse {}/{}/{} news".format(parsed_num//500, nidx, parsed_num))
                res = relation.corenlp_annotate(client, pp.format_news(news))
                if type(res) == str:
                    continue
                persister.add_json(configs.NEWSPARSE, res)
                real_raw_news.append(news)
                parsed_num += 1
                if parsed_num%500 == 0:
                    break
            if parsed_num%500 != 0:
                raise ValueError("cate not 500 parsed")
            # persister.save_json(configs.RAWNEWS, real_raw_news) # 没有parse error 不需要存原始信息了

elif configs.MODE == "reparse":  # reparse failed data
    print("get raw data..")
    rawnews = persister.load_json(configs.RAWNEWS)
    parseres = persister.read_parse()
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    time.sleep(10)
    relation.reannotate(failed_idxs, configs.NEWSPARSE, rawnews, format_func=pp.format_news)

elif configs.MODE == "preprocess":  # use parsed lemma and preprocess it as lda input
    print("preprocess as lda input..")
    if os.path.exists(configs.NEWSINPUT):
        os.remove(configs.NEWSINPUT)
        print("removed", configs.NEWSINPUT)
    newsparse = persister.read_parse()
    # rawnews = persister.load_json(configs.RAWNEWS)
    for idx, parsed in enumerate(newsparse):
        if type(parsed) == str:
            print("{} no parse result, use nltk lemmatized text instead.".format(idx))
            raise ValueError("unexpected parse error")
            # tmp = pp.format_news(rawnews[idx])
            # handled_text = pp.lemma_texts(tmp)
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
    model_name = configs.NEWSMODEL.format(
        int(best_rate*10), len(topic_word), measure, best_iter, min_df)
    persister.persist_lda(configs.NEWSLDA.format(model_name), terms, best_res, topic_word)
    persister.save_model(model_name, best_model)
    persister.save_model(configs.NEWSVEC.format(min_df), vec_model)
    # lda.print_topics(topic_word, terms, best_res)
    vis.plot_line("news_iter_learning", line_data, list(map(str, max_iter_times)),
                  xlabel="learning decay", ylabel="coherence", legend_title="iter time", title=model_name)

elif configs.MODE == "lda":
    _rate = .7
    _iter = 200
    _num = 4
    measure = "c_v"
    min_df = 1
    model_name = configs.NEWSMODEL.format(int(_rate*10), _num, measure, _iter, min_df)
    print("train", model_name)
    news_input = persister.read_input(configs.NEWSINPUT)
    tf, vec_model = lda.extract_feature(news_input, min_df=min_df)
    terms = np.array(vec_model.get_feature_names())
    doc_topic, lda_model = lda.do_lda(tf, topic_num=_num, max_iter=_iter, learning_decay=_rate)
    persister.persist_lda(configs.NEWSLDA.format(model_name), terms, doc_topic, lda_model.components_)
    persister.save_model(model_name, lda_model)
    persister.save_model(configs.NEWSVEC.format(min_df), vec_model)
    print("Done")
else:
    print("unknown mode.")
