from newsgroups import get_news_data, CATEGORIES
import preprocess as pp
import lda
import configs
import persister
import relation

if configs.MODE == "load":
    print("load mode..")
    texts = persister.load_json(configs.NEWSDATA)
    newssent = persister.load_json(configs.NEWSSENT)
    terms, doc_topic, topic_word = persister.read_lda(configs.NEWSLDA)
    newssenttoken = persister.load_json(configs.NEWSSENTTOKEN)
elif configs.MODE == "init":
    raw_data = get_news_data(500)
    newsdata = []
    for cate in raw_data:
        newsdata.extend(cate)

    print("convert to sentences..")
    # save sentences 
    newssent = []
    for news in newsdata:
        newssent.append(pp.split2sent(news))
    persister.save_json(configs.NEWSSENT, newssent)

    # tokenize and lemmatize sentence
    newssenttoken = []
    for news in newssent:
        tokenized_lemmatized_news = []
        for sent in news:
            tokenized_lemmatized_news.append(relation.lemmatize_sent_words(sent))
        newssenttoken.append(tokenized_lemmatized_news)
    persister.save_json(configs.NEWSSENTTOKEN, newssenttoken)

    # print("semantic analysis..")

    print("preprocess data..")
    texts = [' '.join(pp.preprocess_abstract(a)) for a in newsdata]
    persister.save_json(configs.NEWSDATA, newsdata)

    print("run lda..")
    terms, doc_topic, topic_word, perplexity = lda.do_lda(
        texts, topic_num=len(CATEGORIES))
    persister.persist_lda(configs.NEWSLDA, terms, doc_topic, topic_word)

lda.print_topics(topic_word, terms, doc_topic)
