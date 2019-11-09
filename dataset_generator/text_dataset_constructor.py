import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def _get_rf_most_important_feature_indices(data, labels, feature_names, feature_count):
    model = RandomForestClassifier(n_estimators=data.shape[1]//50)
    model.fit(data, labels)
    importance = pd.Series(model.feature_importances_, index=feature_names)
    return np.where(np.isin(feature_names, importance.nlargest(feature_count).index))[0]


def construct_text_dataset(texts, labels, max_df=0.8, feature_selection=True, feature_count=2500):

    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, stop_words='english')
    data_tfidf = tfidf_vectorizer.fit_transform(texts)

    labels = np.array(labels)
    if feature_selection:
        important_feature_indices = _get_rf_most_important_feature_indices(
            data_tfidf, labels, tfidf_vectorizer.get_feature_names(), feature_count)
    else:
        important_feature_indices = np.arange(0, data_tfidf.shape[1])

    data = np.column_stack((data_tfidf[:, important_feature_indices].toarray(), labels))
    feature_names = np.array(tfidf_vectorizer.get_feature_names())[important_feature_indices]
    feature_names = np.append(feature_names, 'Label').astype('str')

    return feature_names, data


def generate_csv(name, x, y):
    data = np.column_stack((x, y))
    np.savetxt("../../../../tmp/" + name, data, delimiter=',')
