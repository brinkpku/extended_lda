# !/usr/bin/env python3
""" visualize some statistic data
"""
import matplotlib.pyplot as plt
import pyLDAvis

import lda
import configs
import persister


def plot_line(line_data, labels):
    """ plot line
    line_data: list of list of tuple, [line1[(x,y),...],...]
    labels: list of str, label for each line
    """
    plt.figure(figsize=(12, 8))
    for idx, l in enumerate(line_data):
        plt.plot([x[0] for x in l], [y[1] for y in l], label=labels[idx])
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()


def pyLDA(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency):
    """ use pyldavis show results in browser
    topic_term_dists : array-like, shape (`n_topics`, `n_terms`)
        Matrix of topic-term probabilities. Where `n_terms` is `len(vocab)`.
    doc_topic_dists : array-like, shape (`n_docs`, `n_topics`)
        Matrix of document-topic probabilities.
    doc_lengths : array-like, shape `n_docs`
        The length of each document, i.e. the number of words in each document.
        The order of the numbers should be consistent with the ordering of the
        docs in `doc_topic_dists`.
    vocab : array-like, shape `n_terms`
        List of all the words in the corpus used to train the model.
    term_frequency : array-like, shape `n_terms`
        The count of each particular term over the entire corpus. The ordering
        of these counts should correspond with `vocab` and `topic_term_dists`.
    """
    data = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
    pyLDAvis.show(data)


if __name__=="__main__":
    news_input = persister.read_input(configs.NEWSINPUT)
    param = lda.generate_lda_parameter(2, 8, 2, [20])
    tf, _ = lda.extract_feature(news_input)
    model = lda.gridsearchCV(param, tf)
    n_topics = list(range(2, 8, 2))
    score = 'mean_test_score'
    log_likelyhoods_5 = [round(s)
                        for s in model.cv_results_[score][:len(n_topics)]]
    log_likelyhoods_7 = [round(s) for s in model.cv_results_[
        score][len(n_topics):2*len(n_topics)]]
    log_likelyhoods_9 = [round(s) for s in model.cv_results_[
        score][2*len(n_topics):3*len(n_topics)]]

    # Show graph
    plot_line([zip(n_topics, log_likelyhoods_5)], ["0.5"])
