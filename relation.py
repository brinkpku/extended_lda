# !/usr/bin/env python3

"""
find semantic relation from documents
"""
import re
import os
from string import punctuation

import utils
import configs
import preprocess as pp
import persister

import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from stanza.server import CoreNLPClient
from sklearn.preprocessing import MinMaxScaler

# corenlp api
if configs.USECLI:
    CLI = StanfordCoreNLP('http://localhost', port=9001)
    # CLI = StanfordCoreNLP(r'/mnt/d/stanford-corenlp-full-2018-02-27', port=9001)
    print("inited stanford CoreNLP client, dont forget to close it!")
else:
    CLI = None
    print("not use stanford CoreNLP client!")

# clean up
def close_cli():
    CLI.close()
    print("closed stanford CoreNLP client!")


def corenlp_annotate(cli, text):
    """ use corenlp annotate(self defined pipeline) sentence
    cli: stanza corenlp client
    text: str, sentences
    return: obj or str, json res if succeeded else error msg
    """
    tmp = 0
    errstr = ""
    while tmp < configs.MAX_TRY:
        tmp += 1
        try:
            res = cli.annotate(text)
            return res.json()
        except Exception as err:
            print("try {}, err: {} {}.".format(tmp, err, len(text)))
            errstr = "err: {} {}.".format(err, len(text))
    return errstr


def reannotate(rerunidxs, persist_file, raw_texts, format_func=pp.format_news):
    """ re annotate failed data
    rerunidxs: list of int, sorted failed idx
    persist_file: str, old persist file
    raw_texts: list of str, raw data
    """
    os.rename(persist_file+".json", persist_file+".bak")
    with CoreNLPClient(properties="./corenlp_server.props", timeout=40000, memory='4G', max_char_length=500000) as client:
        with open(persist_file+".bak") as f:
            for idx, l in enumerate(f):
                if rerunidxs and idx == rerunidxs[0]:
                    print("rennotate", idx)
                    raw_text_idx = rerunidxs.pop(0)
                    res = corenlp_annotate(client, format_func(raw_texts[raw_text_idx]))
                    persister.add_json(persist_file, res)
                else:
                    print("copy", idx, "data")
                    persister.add_input(persist_file+".json", l.strip())


def get_parse_failed_idx(persist_file_name):
    """ get parse failed data index
    persist_file_name: str, parsed res persist file name
    return: list, indexs
    """
    parseres = persister.read_parse(persist_file_name)
    failed_idxs = []
    for idx, i in enumerate(parseres):
        if type(i) == str:
            failed_idxs.append(idx)
    return failed_idxs


# relation extraction
# extract sentence
def lemmatize_sent_words(sent):
    """ word_tokenize sentences(use corenlp) and lemmatize(use nltk)
    return: list of str
    """
    return [pp.lemmatize(word) for word in CLI.word_tokenize(sent)]


def word_in_sents_idx(lemmatized_sents_words, topic_word):
    """ get idxs of sentences contain topic word
    从文本中获取包含“某主题词”的句子的索引
    use extract_important_sents to get more meaning for result
    lemmatized_sents_words: list, [[lemmatized_word,..],[...]]
    return: set of int, indexs
    """
    idxs = set()
    for idx, sent in enumerate(lemmatized_sents_words):
        if topic_word in sent:
            idxs.add(idx)
    return idxs


def get_word_idx(topic_word, sent):
    """ 获取词语在句子中的索引位置
    topic_word: str
    sent: list, tokenized sent
    return: list of int, word index
    """
    res = []
    for idx, i in enumerate(sent):
        if i == topic_word:
            res.append(idx)
    return res


def sent_contained_word_idxs(sent, topic_words):
    """ get idxs of topic words which are contained in the sent
    idx is the index of word in topic_words
    获取句子中所包含的主题词的索引
    sent: list of str, [lemmatized_word,..]
    topic_words: list of str
    return: set of int, indexs
    """
    idxs = set()
    for idx, w in enumerate(topic_words):
        if w in sent:
            idxs.add(idx)
    return idxs


SCALER = MinMaxScaler()


def min_max(raw_data):
    """https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn-preprocessing-minmaxscaler
    raw_data: {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
    """
    return SCALER.fit_transform(raw_data)


def get_lemma_sent(sent_tokens):
    """ get lemma sentence from corenlp tokens
    sent_tokens: list of dict, corenlp annotated sentence tokens result
    return: list of str, [lemma_word, ...]
    """
    tmp = []
    for w in sent_tokens:
        if w["originalText"] in punctuation: # avoid corenlp trans "(" to "-lrb-"
            tmp.append(w["originalText"])
        else:
            tmp.append(w["lemma"])
    return tmp


def convert_parse2lemma_sents(parsed):
    """ convert parse results to lemma sentences, like preprocess.convert_parse2lda_input
    parsed: dict, stanford corenlp annotated result, use its "sentences" value.
    return: list of list of str, lemmatized text
    """
    res = []
    for sent in parsed["sentences"]:
        tmp = get_lemma_sent(sent["tokens"])
        res.append(tmp)
    return res


def extract_important_sents(sents, topic_words, components_values):
    """ extract important sentences idx from a news/abstract by evaluating topic word
    from less important to more important
    sents: list, [[lemmatized_word,..],[...]]
    topic_words: list of str
    components_values: np.array, [can be viewed as pseudocount that represents the number of times word
                                was assigned to topic ], need to normalize
    return: tuple. (np.array, importance, count), important sentences idx(sorted), importance, count
    """
    count = []
    importance = []
    normalized_components_values = min_max([[c] for c in components_values])
    for sent in sents:
        contained_word_idxs = sent_contained_word_idxs(sent, topic_words)
        # sum normalized components value as importance
        importance.append(sum([normalized_components_values[i][0] for i in contained_word_idxs]))
        count.append(len(contained_word_idxs))
    return np.argsort(importance)[-1::-1], importance, count
    

# extract word relation
def extract_word_relation_from_sent(word_idx, sent_deps):
    """ 废弃，现在使用get_governor_relation和get_dependent_relation. 
    get all dependency relation of topic word from one sentence
    word_idx: int, index of topic word in sentence
    sent_deps: list of dict, corenlp enhancedPlusPlusDependencies reslut of one sent
    return: list of dict, dependencies
    """
    word_idx += 1 # root node is 0
    deps = []
    for dep in sent_deps:
        if word_idx == dep[utils.DEPENDENT]:
            deps.append(dep)
        elif word_idx == dep[utils.GOVERNOR]:
            deps.append(dep)
    return deps


def get_governor_relation(word_idx, sent_deps):
    """ 提取指定词语支配的所有关系。
    word_idx: int, index of topic word in sentence
    sent_deps: list of dict, corenlp enhancedPlusPlusDependencies reslut of one sent
    return: list of dict, govern dependencies
    """
    word_idx += 1 # root node is 0
    deps = []
    for dep in sent_deps:
        if word_idx == dep[utils.GOVERNOR]:
            deps.append(dep)
    return deps


def get_dependent_relation(word_idx, sent_deps):
    """ 获取指定词语的依赖关系，与get_governor_relation组合得到全部依存关系。
    get word dependency by real index
    word_idx: int, word index in sentence
    sent_deps: list of dict, corenlp enhancedPlusPlusDependencies reslut of one sent
    return: dict, word dependency
    """
    word_idx += 1 # root node is 0
    for dep in sent_deps:
        if dep[utils.DEPENDENT] == word_idx:
            return dep


def convert_relation2str(dep):
    """ convert dep parse result to string
    dep: dict, dep parse result
    return: str
    """
    return utils.DEP2STR.format(dep[utils.GOVERNORGLOSS], dep[utils.DEP], dep[utils.DEPENDENTGLOSS])


def word_extend_by_pattern(word_idx, sent_tokens):
    """ use pos pattern to extend words, usually for subj and obj
    word_idx: int, word index in sent
    return: tuple of int, extend word index
    sent_tokens: list of dict, corenlp annotated sentence tokens result
    return: tuple of int, extended word indexes in sent
    """
    pos = [i["pos"] for i in sent_tokens]
    left = right = word_idx
    is_nn = pos[word_idx].startswith("N")
    while left>0:
        left -= 1
        if is_nn:
            if pos[left].startswith("N"):
                continue
            elif pos[left].startswith("J"):
                is_nn = False
            else:
                left += 1
                break
        else:
            if pos[left].startswith("J"):
                continue
            else:
                left += 1
                break
    is_nn = pos[word_idx].startswith("N")
    if right == len(sent_tokens)-1: # 恰好是最后一个词，需要+1
        right += 1
    while right+1<len(sent_tokens): # 一般只向前搜索
        right += 1
        if is_nn and pos[right].startswith("N"):
            continue
        else:
            break
    return tuple(range(left, right))


def generate_triples(subjs, predicates, objs):
    """ use permutation and combination to generate triples
    params: list of tuple
    return: list of list of tuple
    """
    triples = []
    for s in subjs:
        for p in predicates:
            if not objs:
                triples.append([s, p, None])
                continue
            for o in objs:
                triples.append([s, p, o])
    return triples


def extract_triples_from_sent(sent_deps, sent_tokens):
    """ extract triples from a sent
    sent_deps: list of dict, corenlp enhancedPlusPlusDependencies reslut of one sent
    sent_tokens: list of dict, corenlp annotated sentence tokens result
    """
    root_idx = sent_deps[0][utils.DEPENDENT] - 1 # real idx in sentence, minus root node(-1)
    root_relations = get_governor_relation(root_idx, sent_deps)
    triples = []
    subjs = []
    predicate = [(root_idx, )]
    objs = []
    advcls = []
    xcomps = []
    for r in root_relations:
        if r[utils.DEP].startswith("nsubj"): # 主句主语
            subjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
        elif r[utils.DEP] == "dobj": # 主句直接宾语
            objs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
        elif r[utils.DEP].startswith("nmod"): # 介词短语
            # TODO find way save nmod info
            objs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
        elif r[utils.DEP].startswith("advcl"): # 状语从句
            advcls.append(r)
        elif r[utils.DEP].startswith("xcomp"): # 开放补语从句
            xcomps.append(r)
    triples.extend(generate_triples(subjs, predicate, objs))
    # clause extension，暂时只扩展一层从句
    for pr in advcls: # 状语从句
        clause_predicate_idx = pr[utils.DEPENDENT] - 1 # 从句谓词句子中索引
        clause_relations = get_governor_relation(clause_predicate_idx, sent_deps)
        csubjs = []
        cpredict = [(clause_predicate_idx, )]
        cobjs = []
        for r in clause_relations:
            if r[utils.DEP].startswith("nsubj"): # 主句主语
                csubjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP] == "dobj": # 主句直接宾语
                cobjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP].startswith("nmod"): # 介词短语
                nmod = re.findall("nmod:(.+)", r[utils.DEP])
                if nmod: # TODO find way save nmod info
                    pass
                cobjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
        if not csubjs: # 从句没有主语，使用上一层主语
            csubjs = subjs
        triples.extend(generate_triples(csubjs, cpredict, cobjs))
    for xr in xcomps: # 开放补语从句
        clause_predicate_idx = xr[utils.DEPENDENT] - 1 # 从句谓词句子中索引
        clause_relations = get_governor_relation(clause_predicate_idx, sent_deps)
        csubjs = []
        cpredict = [(clause_predicate_idx, )]
        cobjs = []
        for r in clause_relations:
            if r[utils.DEP].startswith("nsubj"): # 主句主语
                csubjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP] == "dobj": # 主句直接宾语
                cobjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP].startswith("nmod"): # 介词短语
                nmod = re.findall("nmod:(.+)", r[utils.DEP])
                if nmod: # TODO find way save nmod info
                    pass
                cobjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
        if not csubjs: # 从句没有主语，使用上一层主语
            csubjs = subjs
        triples.extend(generate_triples(csubjs, cpredict, cobjs))
    return triples


NOUN_PHRASE = "J*[NF]+"

@utils.timer
def get_phrases_by_pattern(parses):
    """ 
    parses: list of dict, corpus parsed results
    return: list of str
    """
    phrases = []
    for parse in parses:
        if type(parse) == str:
            continue
        for sent in parse["sentences"]:
            pos = [i["pos"] for i in sent["tokens"]]
            words = [i["lemma"] for i in sent["tokens"]]
            _s = _e = 0
            extension = False
            for idx, p in enumerate(pos):
                if extension:
                    if p.startswith(("N", "F")):
                        _e = idx+1
                    elif _s < _e-1: # match phrase
                        phrases.append(words[_s:_e])
                        extension = False # rest window status
                    else:
                        extension = False
                else:
                    if p.startswith(("J", "N", "F")):
                        _s = idx
                        extension = True
    return phrases



if __name__ == '__main__':
    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    # print('Tokenize:', CLI.word_tokenize(sentence))
    # print('Part of Speech:', CLI.pos_tag(sentence))
    # print('Named Entities:', CLI.ner(sentence))
    # print('Constituency Parsing:', CLI.parse(sentence))
    # print('Dependency Parsing:', CLI.dependency_parse(sentence))
    # close_cli()
    
    import newsgroups
    from stanza.server import CoreNLPClient
    news = newsgroups.get_news_data()
    with CoreNLPClient(properties="./corenlp_server.props", timeout=30000, memory='5G') as client:
        n = pp.format_news(news[0][0])
        print(corenlp_annotate(client, n))
