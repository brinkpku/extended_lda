# !/usr/bin/env python3
import os

MAX_TRY = 3


MODE = os.getenv("mode", "init") # init, load, parse, reparse, preprocess, lda
RECOVERIDX = os.getenv("recoveridx", 0)
USECLI = os.getenv("use_cli", False)

# json, raw data
RAWNEWS = "rawnews"
RAWABSTRACT = "rawabstract"

# json, preprocessed data, used as lda params
NEWSINPUT = "news_input.lemma"
ABSTRACTINPUT = "abstract_input.lemma"

# json, sentence data
# only persist parse data, sentence data can be generated from parse data
NEWSSENT = "newssent"
ABSTRACTSENT = "abstractsent"
NEWSSENTTOKEN = "newssenttoken"  # lemmatized sentence words
ABSTRACTSENTTOKEN = "abstractsenttoken"

NEWSPARSE = "newsparse"  # corenlp analyse results
ABSTRACTPARSE = "abstractparse"

# lda model, .model
NEWSMODEL = "newslda"
ABSTRACTMODEL = "abstractlda"
MODELPATH = "models/"

# numpy, lda result: terms, doc_topic, topic_word
ABSTRACTLDA = "abstract_lda"
NEWSLDA = "news_lda"
