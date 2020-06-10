# !/usr/bin/env python3
""" visualize some statistic data
"""
import matplotlib.pyplot as plt

import lda
import configs
import persister


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
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()
