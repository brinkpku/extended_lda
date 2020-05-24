# !/usr/bin/env python3
import os

MAX_TRY = 3


MODE = os.getenv("mode", "init") # init, load, recover
RECOVERIDX = os.getenv("recoveridx", 0)
USECLI = os.getenv("use_cli", True)

# json, raw data
RAWNEWS = "rawnews"
RAWABSTRACT = "rawabstract"

# json, preprocessed data, used as lda params
NEWSINPUT = "news_input"
ABSTRACTINPUT = "abstract_input"

# json, sentence data
NEWSSENT = "newssent"
ABSTRACTSENT = "abstractsent"
NEWSSENTTOKEN = "newssenttoken"  # lemmatized sentence words
ABSTRACTSENTTOKEN = "abstractsenttoken"
NEWSPARSE = "newsparse"  # corenlp analyse results


# numpy, lda result: terms, doc_topic, topic_word
ABSTRACTLDA = "abstract_lda"
NEWSLDA = "news_lda"
