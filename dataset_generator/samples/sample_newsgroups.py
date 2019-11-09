import numpy as np
import pandas as pd
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from dataset_generator.text_dataset_constructor import construct_text_dataset

categories = ['alt.atheism', 'soc.religion.christian']
news = fetch_20newsgroups(subset='all', categories=categories)

feature_names, data = construct_text_dataset(
    texts=news.data, labels=news.targets, feature_selection=True, feature_count=3000)

df = pd.DataFrame(data=data, columns=feature_names)
df.to_csv('../../data/20_newsgroups_3000_features.csv')
