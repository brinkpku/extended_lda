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
NEWSSENTTOKEN = "newssenttoken"  # lemmatized sentence words
ABSTRACTSENTTOKEN = "abstractsenttoken"
NEWSPARSE = "newsparse"  # corenlp analyse results


# numpy, lda result: terms, doc_topic, topic_word
ABSTRACTLDA = "abstract_lda"
NEWSLDA = "news_lda"
