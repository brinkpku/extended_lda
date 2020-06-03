# !/usr/bin/env python3
import json
import time

from stanza.server import CoreNLPClient

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
elif configs.MODE == "init":
    print("get raw data..")
    rawnews = persister.load_json(configs.RAWNEWS)
    #raw_data = get_news_data(500)
    #rawnews = []
    # for cate in raw_data:
    #    rawnews.extend(cate)
    #persister.save_json(configs.RAWNEWS, rawnews)

    # print("semantic analysis..")
    # print("annotate sentence..")
    # with CoreNLPClient(properties="./corenlp_server.props", timeout=30000, memory='4G') as client:
    #     for idx, news in enumerate(rawnews):
    #         if idx < int(configs.RECOVERIDX):
    #             print("recover", idx)
    #             continue
    #         print("parse {}/{} news".format(idx, len(rawnews) - 1))
    #         res = relation.corenlp_annotate(client, news)
    #         persister.add_json(configs.NEWSPARSE, res)
    newsparse = persister.read_parse()

    print("preprocess as lda input..")
    for idx, parsed in enumerate(newsparse):
        if type(parsed) == str:
            print("{} no parse result, use raw text instead of lemmatized.".format(idx))
            tmp = pp.format_news(rawnews[idx])
            handled_text = " ".join(pp.preprocess_abstract(tmp))
        else:
            print("convert to lda input:{}/{}".format(idx, len(newsparse) - 1))
            handled_text = " ".join(
                [" ".join([w["lemma"] for w in sent["tokens"]]) for sent in parsed["sentences"]])
        preprocessed = pp.preprocess_abstract(handled_text)
        persister.add_input(configs.NEWSINPUT, " ".join(preprocessed))
    lda_input = persister.read_input(configs.NEWSINPUT)

    print("run lda..")
    terms, doc_topic, topic_word, perplexity = lda.do_lda(
        lda_input, topic_num=len(CATEGORIES))
    persister.persist_lda(configs.NEWSLDA, terms, doc_topic, topic_word)

elif configs.MODE == "rerun":
    print("get raw data..")
    rawnews = persister.load_json(configs.RAWNEWS)
    parseres = persister.read_parse()
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    time.sleep(10)
    relation.reannotate(failed_idxs, configs.NEWSPARSE, rawnews)


lda.print_topics(topic_word, terms, doc_topic)
