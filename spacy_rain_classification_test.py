import pandas as pd
import spacy
import numpy as np
df = pd.read_csv('test.csv')
print("Loading from /content/model/")
nlp2 = spacy.load('italian_rain_model')

data={}
res=[]
for index, row in df.iterrows():
  if type(row[3]) == float and np.isnan(row[3]):
    print('error record')
  else:
    docs = nlp2(row[3])
    data = {'text': row[3], 'POSITIVE': docs.cats['POSITIVE'], 'NEGATIVE': docs.cats['NEGATIVE']}
    res.append(data)


finaldf = pd.DataFrame(data=res, columns=['text','POSITIVE','NEGATIVE'])
finaldf.to_csv('italian_result')

