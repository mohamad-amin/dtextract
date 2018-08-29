import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def construct_text_dataset(texts, labels, max_df=0.9):

    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, stop_words='english')
    data_tfidf = tfidf_vectorizer.fit_transform(texts)

    labels = np.array(labels)
    data = np.column_stack((data_tfidf.toarray(), labels))
    feature_names = tfidf_vectorizer.get_feature_names()
    feature_names.append('Label')
    feature_names = np.array(feature_names).astype('str')

    return feature_names, data
