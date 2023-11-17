#Import Dependencies 
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import re

#Load Model 
model = load_model(r'SentimentModel.h5')

#Define the website to scrape from
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AAPL']

#Scraping financial headlines
news_tables = {}
for ticker in tickers:
    
    url = finviz_url + ticker
    
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

#Parsing the data into a dataframe
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        parsed_data.append([ticker, title])

df = pd.DataFrame(parsed_data, columns=['ticker','title'])

#Process the scraped data into numerical vector
df = df['title'].apply(str.lower)
df = df.apply(lambda x: re.sub('[^a-zA-Z0-9\s]',"",x))
tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(df.values)
X = tokenizer.texts_to_sequences(df.values)
X = pad_sequences(X)

#Padding the vector to fit the model 
padding_size = 50 - len(X[0])
padding_vector = np.zeros(padding_size)
to_predict = []
for row in X:
    concatenated_row = np.concatenate((padding_vector, row))
    concatenated_row = np.array(concatenated_row)
    to_predict.append(concatenated_row)
predict_news = pd.DataFrame(np.array(to_predict))

#Load the model and predict sentiments 
model = load_model(r'SentimentModel.h5')
predictions = model.predict(predict_news)
length = len(predictions)

#Convert the numerical sentiments into text format 
def find_highest_position(arr):
    highest_num = max(arr)
    for index, num in enumerate(arr):
        if num == highest_num:
            return index
        
def convertToSentiment(prediction):
    if prediction == [0, 1, 0]:
        return 'neutral'
    elif prediction == [0, 0, 1]:
        return 'positive'
    elif prediction == [1, 0, 0]:
        return 'negative'

def format_predictions(predictions):
    format_predictions = []
    for prediction in predictions:
        position = find_highest_position(prediction)
        prediction = [0] * len(prediction)
        prediction[position] = 1
        convert_prediction = convertToSentiment(prediction)
        format_predictions.append(convert_prediction)
    return format_predictions

#Combining the news headlines with sentiments 
predictions = format_predictions(predictions)
predictions = np.array(predictions)
predictions = pd.DataFrame(predictions, columns=['sentiments'])
combined_df = pd.concat([df, predictions], axis=1, join='inner')
print(combined_df)

#Export to .csv format 
combined_df.to_csv('sentiments.csv', index=False)