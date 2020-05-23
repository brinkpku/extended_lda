# !/usr/bin/env python3
from sklearn.datasets import fetch_20newsgroups
import numpy as np


CATEGORIES = ['comp.graphics', 'comp.windows.x',
              'rec.autos', 'rec.sport.baseball']
SIZE = 10


def get_news_data(size=SIZE, categories=CATEGORIES):
    """
    get 20newsgroups data
    size: int, news num of each category
    categories: list of str, news categories
    return: list, [[cate1,..],[cate2,..]..]
    """
    newsdata = fetch_20newsgroups(subset="all", categories=categories, remove=(
        "footers", "qouties", "headers"), shuffle=False)
    selected_news = []
    for idx, _ in enumerate(newsdata.target_names):
        filtered_idx = np.argwhere(newsdata.target == idx)[:size]
        selected = [newsdata.data[nidx[0]] for nidx in filtered_idx]
        selected_news.append(selected)
    return selected_news


if __name__=="__main__":
    tmp = get_news_data(500)
    newsdata = []
    for cate in tmp:
        newsdata.extend(cate)
    print(newsdata[1200])
    from string import punctuation
    print(punctuation)
