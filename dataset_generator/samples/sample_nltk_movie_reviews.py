import random
import pandas as pd

from nltk.corpus import movie_reviews
from dataset_generator.text_dataset_constructor import construct_text_dataset

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
df = pd.DataFrame(documents)

class_names = {'neg': 0, 'pos': 1}
df[0] = df[0].map(lambda x: ' '.join(x))
df[1] = df[1].map(lambda x: class_names[x])

feature_names, data = construct_text_dataset(
    texts=df[0].tolist(), labels=df[1].tolist(), feature_selection=True, feature_count=3000)

df = pd.DataFrame(data=data, columns=feature_names)
df.to_csv('../../data/nltk_movie_reviews_3000_features.csv')
