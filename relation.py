# !/usr/bin/env python3

"""
find semantic relation from documents
"""
import re
import os
from string import punctuation
from functools import reduce

import utils
import configs
import preprocess as pp
import persister
import evaluate
import embedding

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


NMOD = ["in", "at", "from", "through", "to", "on", "of", "about", "over", "for", "as", "with", "without"]
def word_extend_by_relation(word_idx, sent_deps, extend_nmod=False):
    """ use relation to extend words, usually for subj and obj
    simply, nmod:xx only extend once
    return: tuple of int, extended word indexes in sent
    """
    res = [word_idx,]
    word_relations = get_governor_relation(word_idx, sent_deps)
    for r in word_relations:
        if r[utils.DEP] == "compound" or r[utils.DEP] == "amod":
            res.append(r[utils.DEPENDENT]-1)
        elif r[utils.DEP].startswith("nmod"):
            nmod = re.findall("nmod:(.+)", r[utils.DEP])
            if nmod and nmod[0] in NMOD:
                secondary_relations = get_governor_relation(r[utils.DEPENDENT]-1, sent_deps)
                for sr in secondary_relations:
                    if sr[utils.DEP] == "case":
                        res.append(sr[utils.DEPENDENT]-1) # 介词
                        res.append(r[utils.DEPENDENT]-1) # 修饰的名词
                        break
        elif r[utils.DEP].startswith("conj") or r[utils.DEP] == "cc": # 并列关系
            res.append(r[utils.DEPENDENT] - 1)
        elif extend_nmod and r[utils.DEP] == "case":
            res.append(r[utils.DEPENDENT] - 1)
    return tuple(sorted(res))


def predicate_extend(word_idx, sent_deps, predicate):
    """ 原地扩展动词为 动词+介词 或 动词并列形式
    """
    extend_nmod = True
    res = [word_idx,]
    word_relations = get_governor_relation(word_idx, sent_deps)
    for r in word_relations:
        if r[utils.DEP].startswith("nmod"):
            nmod = re.findall("nmod:(.+)", r[utils.DEP])
            if nmod and nmod[0] in NMOD:
                secondary_relations = get_governor_relation(r[utils.DEPENDENT]-1, sent_deps)
                for sr in secondary_relations:
                    if sr[utils.DEP] == "case" and sr[utils.DEPENDENT] == r[utils.GOVERNOR]+1:#动词介词相邻
                        res.append(sr[utils.DEPENDENT]-1)
                        extend_nmod = False # 谓词已经扩展介词，宾语不需要再扩展
                        break
        elif r[utils.DEP] == "compound:prt":
            res.append(r[utils.DEPENDENT] - 1)
        # elif r[utils.DEP].startswith("conj") or r[utils.DEP] == "cc": # 并列关系
        #     res.append(r[utils.DEPENDENT] - 1)
    predicate[0] = tuple(sorted(res))
    return extend_nmod


def tuple_contain(longer, shorter):
    """ judge one tuple whether contains another tuple
    """
    if len(longer) < len(shorter):
        longer, shorter = shorter, longer
    l = s = 0
    match = False
    while l<len(longer):
        if s == len(shorter):
            return True
        if longer[l] == shorter[s]:
            if not match:
                match = True
        else:
            if match:
                match = False
            s = 0
        l += 1
        s += 1
    return False
        

def combine_words(lst_of_tuple):
    """ 合并存在包含关系的成分
    """
    mask = [True]*len(lst_of_tuple)
    idx = 0
    while idx<len(lst_of_tuple):
        iidx = idx + 1
        while iidx < len(lst_of_tuple):
            if tuple_contain(lst_of_tuple[idx], lst_of_tuple[iidx]):
                if len(lst_of_tuple[idx]) >= len(lst_of_tuple[iidx]):
                    mask[iidx] = False
                else:
                    mask[idx] = False
            iidx += 1
        idx +=1
    return [t for idx, t in enumerate(lst_of_tuple) if mask[idx]]      


def generate_triples(subjs, predicates, objs):
    """ use permutation and combination to generate triples
    params: list of tuple
    return: list of list of tuple
    """
    triples = []
    subjs = combine_words(subjs)
    predicates = combine_words(predicates)
    objs = combine_words(objs)
    for s in subjs:
        for p in predicates:
            if not objs:
                triples.append([s, p, None])
                continue
            for o in objs:
                triples.append([s, p, o])
    return triples


def extract_triples_from_sent(sent_deps, sent_tokens, use_relation=False):
    """ extract triples from a sent
    sent_deps: list of dict, corenlp enhancedPlusPlusDependencies reslut of one sent
    sent_tokens: list of dict, corenlp annotated sentence tokens result
    use_relation: bool, decide word extend method
    """
    triples = []
    root_idx = sent_deps[0][utils.DEPENDENT] - 1 # real idx in sentence, minus root node(-1)
    roots = [root_idx,] # 句子支配词，用于迭代主句和从句
    last_subjs = []
    in_clause = False
    while roots:
        predicate_idx = roots.pop(0) # 用队列维护句子层次
        subjs = []
        predicate = [(predicate_idx, )]
        objs = []
        root_relations = get_governor_relation(predicate_idx, sent_deps)
        not_cop = True
        for r in root_relations:
            if r[utils.DEP].startswith("nsubj"): # 主句主语
                if use_relation:
                    subjs.append(word_extend_by_relation(r[utils.DEPENDENT]-1, sent_deps))
                else:
                    subjs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP] == "dobj": # 主句直接宾语
                if use_relation:
                    objs.append(word_extend_by_relation(r[utils.DEPENDENT]-1, sent_deps))
                else:
                    objs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP] == "cop":
                not_cop = False
                predicate = [(r[utils.DEPENDENT] - 1, )]
                # 主系表结构中支配词实际为表语
                if use_relation:
                    objs.append(word_extend_by_relation(r[utils.GOVERNOR]-1, sent_deps))
                else:
                    objs.append(word_extend_by_pattern(r[utils.GOVERNOR]-1, sent_tokens))
            elif r[utils.DEP].startswith("nmod"): # 介词短语
                nmod = re.findall("nmod:(.+)", r[utils.DEP])
                if nmod and nmod[0] in NMOD and not_cop: 
                    # 当主系表结构当前支配词并非动词，不需要扩展；其介词关系在cop中扩展
                    extend_nmod = predicate_extend(predicate_idx, sent_deps, predicate)
                    if use_relation:
                        objs.append(word_extend_by_relation(r[utils.DEPENDENT]-1, sent_deps, extend_nmod))
                    else:
                        objs.append(word_extend_by_pattern(r[utils.DEPENDENT]-1, sent_tokens))
            elif r[utils.DEP].startswith(("advcl", "xcomp", "ccomp")): # 状语从句、开放补语从句、补语从句
                roots.append(r[utils.DEPENDENT] - 1)
                last_subjs.append(subjs) # 将上一层主语列表的引用存储在队列中，同步更新上一层主语信息
        if in_clause:
            last_subj = last_subjs.pop(0) # 队列维护上一层主语信息，与从句信息一一对应
            if not subjs: # 从句没有主语，使用上一层主语
                subjs = last_subj
        triples.extend(generate_triples(subjs, predicate, objs))
        in_clause = True # 除了第一次循环，其余都是从句
    return triples


def evaluate_topic_triple(lemma_triple, topic_words, components_values):
    """ caculate importance of triple for topic. metrics like extract_important_sents
    lemma_triple: list of str, lemmatized words of triple
    return: float, importance score
    """
    importance = .0
    normalized_components_values = min_max([[c] for c in components_values])
    for lemma in lemma_triple:
        if lemma in topic_words:
            idx = topic_words.index(lemma)
            importance += normalized_components_values[idx][0]
    return importance


def score_tf(lemma_triple, frequency, vec_tf):
    """ 三元组中词语在该话题top文章中出现的频次之和
    """
    idxs = evaluate.filter_triple2term_idx(lemma_triple, vec_tf)
    return sum([frequency[i] for i in idxs])


def score_tfidf(lemma_triple, tfidf, vec_tfidf):
    idxs = evaluate.filter_triple2term_idx(lemma_triple, vec_tfidf)
    return sum([tfidf[i] for i in idxs])


def score_w2v_dist(lemma_triple, doc_w2v, model):
    triple_w2v = embedding.get_doc_dense_vec(model, 
    " ".join([lemma for argument in lemma_triple for lemma in argument]))
    if type(triple_w2v) is bool:
        return .0
    else:
        return np.dot(triple_w2v, doc_w2v)/(np.linalg.norm(triple_w2v)*(np.linalg.norm(doc_w2v))) # cosine


def extend_lda_results(parse_results, input_text, top_terms, top_docs, terms, res_format="originalText", top_n=-1, score_method="basic", is_news=False):
    """ extend lda result to triples by relation parse and information extraction
    parse_results: list of dict, corpus corenlp parse result
    input_text: list of str, docs in list
    top_terms: list of list of tuple, lda get_topics result
    top_docs: list of list of tuple, lda get_topics result
    terms: np.array of str, vocabulary
    score_method: str, 'basic', 'tf', 'itf' 
        basic is simplest, use evaluate_topic_triple
        tf: score_tf
        tfidf: score_tf_idf
        w2v: score_w2v_dist
    return: list of list of tuple, (triple(list of list of str), float score)
    """
    if res_format not in ["originalText", "lemma"]:
        raise ValueError("format support 'originalText' or 'originalText'")
    if is_news:
        w2v_model = persister.load_wv(configs.NEWSWV.format(100))
    else:
        w2v_model = persister.load_wv(configs.ABSWV.format(100))
    all_triples = []
    for t_idx, topic_res in enumerate(top_docs):
        topic_triples = []
        scores = []
        topic_terms = [terms[x[0]] for x in top_terms[t_idx]]
        topic_components = [x[1] for x in top_terms[t_idx]]
        topic_texts = [input_text[top_doc[0]] for top_doc in topic_res]
        topic_term_fre, vec_tf = evaluate.get_topic_term_frequency(topic_texts)
        tfidf, vec_tfidf = evaluate.get_topic_term_tfidf(topic_texts)
        for top_doc in topic_res:
            doc_idx = top_doc[0]
            if type(parse_results[doc_idx]) is str:
                print(doc_idx, "parse", "err")
                continue
            sents = convert_parse2lemma_sents(parse_results[doc_idx])
            sort_idxs, importance, _ = extract_important_sents(sents, 
                [terms[x[0]] for x in top_terms[t_idx]], [x[1] for x in top_terms[t_idx]])
            doc_w2v = embedding.get_doc_dense_vec(w2v_model, input_text[doc_idx])
            for i in sort_idxs:
                if importance[i]>0:
                    sent_tokens = parse_results[doc_idx]["sentences"][i]["tokens"]
                    sent_deps = parse_results[doc_idx]["sentences"][i]["enhancedPlusPlusDependencies"]
                    triples = extract_triples_from_sent(sent_deps, sent_tokens, use_relation=True)
                    for triple in triples:
                        if None in triple:
                            continue
                        extract_res = [
                                        [sent_tokens[i][res_format] for i in triple[0]],
                                        [sent_tokens[i][res_format] for i in triple[1]],
                                        [sent_tokens[i][res_format] for i in triple[2]],
                                    ]
                        topic_triples.append(extract_res)
                        lemma_triple = None
                        if res_format == "lemma":
                            lemma_triple = extract_res
                        else:
                            lemma_triple = [
                                        [sent_tokens[i]["lemma"] for i in triple[0]],
                                        [sent_tokens[i]["lemma"] for i in triple[1]],
                                        [sent_tokens[i]["lemma"] for i in triple[2]],
                                    ]
                        if score_method == "basic": 
                            s = evaluate_topic_triple(reduce(lambda a,b:a+b, lemma_triple), topic_terms, topic_components)
                        elif score_method == "tf":
                            s = score_tf(lemma_triple, topic_term_fre, vec_tf)
                        elif score_method == "tfidf":
                            s = score_tfidf(lemma_triple, tfidf, vec_tfidf)
                        elif score_method == "w2v":
                            s = score_w2v_dist(lemma_triple, doc_w2v, w2v_model)
                        else:
                            raise ValueError("unkown score method")
                        scores.append(s)
        filtered_triples = []
        for idx in np.argsort(scores)[-1::-1]:
            if scores[idx]>0:
                filtered_triples.append((topic_triples[idx], scores[idx]))
            else:
                break
        if top_n < 0:
            all_triples.append(filtered_triples)
        else:
            all_triples.append(filtered_triples[:top_n])
    return all_triples


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
