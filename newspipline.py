# !/usr/bin/env python3
import json
import time

from stanza.server import CoreNLPClient
import numpy as np

from newsgroups import get_news_data, CATEGORIES
import preprocess as pp
import lda
import configs
import persister
import relation

if configs.MODE == "load":
    print("load mode..")
    rawnews = persister.load_json(configs.RAWNEWS)
    lda_input = persister.read_input(configs.NEWSINPUT)
    terms, doc_topic, topic_word = persister.read_lda(configs.NEWSLDA)
    newsparse = persister.read_parse()
elif configs.MODE == "init": # prepare and preprocess data
    print("get raw data..")
    raw_data = get_news_data(500)
    rawnews = []
    for cate in raw_data:
       rawnews.extend(cate)
    persister.save_json(configs.RAWNEWS, rawnews)

elif configs.MODE == "parse":
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

elif configs.MODE == "reparse":
    print("get raw data..")
    rawnews = persister.load_json(configs.RAWNEWS)
    parseres = persister.read_parse()
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    time.sleep(10)
    relation.reannotate(failed_idxs, configs.NEWSPARSE, rawnews)

elif configs.MODE == "preprocess":
    print("preprocess as lda input..")
    newsparse = persister.read_parse()
    for idx, parsed in enumerate(newsparse):
        if type(parsed) == str:
            print("{} no parse result, use raw text instead of lemmatized.".format(idx))
            tmp = pp.format_news(rawnews[idx])
            handled_text = " ".join(pp.preprocess(tmp))
        else:
            print("convert to lda input:{}/{}".format(idx, len(newsparse) - 1))
            handled_text = pp.convert_parse2lda_input(parsed)
        preprocessed = pp.preprocess(handled_text)
        persister.add_input(configs.NEWSINPUT, " ".join(preprocessed))
    news_input = persister.read_input(configs.NEWSINPUT)

elif configs.MODE == "lda":
    print("run lda..")
    tf, vec_model = lda.extract_feature(news_input)
    terms = np.array(vec_model.get_feature_names())
    doc_topic, lda_model = lda.do_lda(tf, topic_num=len(CATEGORIES))
    topic_word = lda_model.components_
    persister.persist_lda(configs.NEWSLDA, terms, doc_topic, topic_word)
    persister.save_model(configs.NEWSMODEL, lda_model)
    lda.print_topics(topic_word, terms, doc_topic)


else:
    print("unknown mode.")

