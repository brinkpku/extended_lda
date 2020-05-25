# !/usr/bin/env python3

import json

import numpy as np

import configs

# persist lda input
def add_input(f_name, line):
    with open(f_name, "a+", encoding="utf8") as f:
        f.write("".join([line, "\n"]))

def read_input(f_name):
    with open(f_name, encoding="utf8") as f:
        lda_input = f.readlines()
    return lda_input

# save json, such as preprocessed result, sentence
def save_json(json_name, obj):
    with open(json_name+".json", "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=True)

def add_json(json_name, line_obj):
    with open(json_name+".json", "a+", encoding="utf8") as f:
        f.write("".join([json.dumps(line_obj, ensure_ascii=True), "\n"]))

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


def read_parse(json_name=configs.NEWSPARSE):
    with open(json_name+".json", encoding="utf8") as f:
        newsparse = f.readlines()
    newsparse = [json.loads(n) for n in newsparse]
    return newsparse

if __name__ == "__main__":
    print("read parse")
    parseres = read_parse()
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    print(failed_idxs)
    print(len(failed_idxs))
    