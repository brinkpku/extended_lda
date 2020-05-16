# !/usr/bin/env python3
import os

# init, load
MODE = os.getenv("mode", "init")

# json, preprocessed data, used as lda params
NEWSDATA = "newsdata"
ABSTRACTDATA = "abstract_data"

# json, sentence data
NEWSSENT = "newssent"
ABSTRACTSENT = "abstractsent"

# numpy, lda result: terms, doc_topic, topic_word
ABSTRACTLDA = "abstract_lda"
NEWSLDA = "news_lda"