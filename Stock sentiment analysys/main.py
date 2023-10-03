import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('data/Data.csv')
df.head(3)

headlines = []

cleaned_df = df.copy()

cleaned_df.replace('[^a-zA-Z]', ' ',regex=True,inplace=True)
cleaned_df.replace('[ ]+', ' ',regex=True,inplace=True)


for row in range(len(df)):
    headlines.append(' '.join(str(x) for x in cleaned_df.iloc[row,2:]).lower())


cv = CountVectorizer(ngram_range=(2,2))
cv.fit(headlines)


headlines[0]

train_data = cleaned_df[df['Date']<'20150101']
test_data = cleaned_df[df['Date']>'20141231']

train_data_len = len(train_data)

train_headlines = cv.transform(headlines[:train_data_len])
test_headlines = cv.transform(headlines[train_data_len:])

rfc = RandomForestClassifier(n_estimators=200,criterion='entropy')

rfc.fit(train_headlines,train_data['Label'])

preds = rfc.predict(test_headlines)

print(accuracy_score(test_data['Label'],preds))
confusion_matrix(test_data['Label'],preds)

joblib.dump(rfc, 'stock_sentiment.pkl')