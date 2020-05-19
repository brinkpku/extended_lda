# !/usr/bin/env python3

import json

import numpy as np

import configs


# save json, such as preprocessed result, sentence
def save_json(json_name, obj):
    with open(json_name+".json", "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=True)

def add_json(json_name, line_obj):
    with open(json_name+".json", "a+", encoding="utf8") as f:
        f.writelines("".join([json.dumps(line_obj, ensure_ascii=True), "\n"]))

# save npz, lda result
def save_npz(npz_name, *args, **kwds):
    np.savez(npz_name, *args, **kwds)


def load_json(json_name):
    with open(json_name+".json", encoding="utf8") as f:
        return json.load(f)


def load_npz(npz_name):
    return np.load(npz_name+".npz")


def persist_lda(npz_name, terms, doc_topic, topic_word):
    save_npz(npz_name, terms=terms, doc_topic=doc_topic, topic_word=topic_word)


def read_lda(npz_name):
    npz = load_npz(npz_name)
    return npz["terms"], npz["doc_topic"], npz["topic_word"]
