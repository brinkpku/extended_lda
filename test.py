# !/usr/bin/env python3
import json

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
    newssent = persister.load_json(configs.NEWSSENT)
    terms, doc_topic, topic_word = persister.read_lda(configs.NEWSLDA)
    newssenttoken = persister.load_json(configs.NEWSSENTTOKEN)
    newsparse = persister.read_parse()
elif configs.MODE == "init":
    print("get raw data..")
    raw_data = get_news_data(500)
    rawnews = []
    for cate in raw_data:
        rawnews.extend(cate)
    persister.save_json(configs.RAWNEWS, rawnews)

    # print("semantic analysis..")
    print("annotate sentence..")
    with CoreNLPClient(properties="./corenlp_server.props", timeout=30000, memory='5G') as client:
        for idx, news in enumerate(rawnews):
            if configs.MODE == "recover" and idx < configs.RECOVERIDX:
                continue
            print("parse {}/{} news".format(idx, len(rawnews) - 1))
            res = relation.corenlp_annotate(client, news)
            persister.add_json(configs.NEWSPARSE, res)
    newsparse = persister.read_parse()

    # print("convert to sentences..")
    # neednt to save sentences

    # tokenize and lemmatize sentence
    # newssenttoken = []
    # for news in newssent:
    #     tokenized_lemmatized_news = []
    #     for sent in news:
    #         tokenized_lemmatized_news.append(
    #             relation.lemmatize_sent_words(sent))
    #     newssenttoken.append(tokenized_lemmatized_news)
    # persister.save_json(configs.NEWSSENTTOKEN, newssenttoken)

    print("preprocess as lda input..")
    for idx, parsed in enumerate(newsparse):
        print("convert to lda input:{}/{}".format(idx, len(newsparse) - 1))
        handled_text = " ".join(
            [" ".join([w["lemma"] for w in sent["tokens"]]) for sent in res["sentences"]])
        preprocessed = pp.preprocess_abstract(handled_text)
        persister.add_input(configs.NEWSINPUT, preprocessed)
    lda_input = persister.read_input(configs.NEWSINPUT)

    print("run lda..")
    terms, doc_topic, topic_word, perplexity = lda.do_lda(
        lda_input, topic_num=len(CATEGORIES))
    persister.persist_lda(configs.NEWSLDA, terms, doc_topic, topic_word)


# lda.print_topics(topic_word, terms, doc_topic)
