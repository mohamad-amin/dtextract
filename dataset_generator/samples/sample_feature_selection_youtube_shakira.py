import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

FEATURE_COUNT = 50

df = pd.read_csv('../../data/youtube_shakira_old.csv')

model = RandomForestClassifier(n_estimators=50)
model.fit(df.iloc[:, :-1], df.iloc[:, -1])

importance = pd.Series(model.feature_importances_, index=df.columns[:-1])
important_features = np.where(np.isin(df.columns, importance.nlargest(FEATURE_COUNT).index))[0]

data = np.column_stack((df.iloc[:, important_features].values, df.iloc[:, -1]))
feature_names = np.append(df.columns[important_features].values, 'Label').astype('str')

df = pd.DataFrame(data=data, columns=feature_names)
df.to_csv('../../data/youtube_shakira.csv')

