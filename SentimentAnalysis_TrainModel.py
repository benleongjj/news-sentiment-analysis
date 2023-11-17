#Import Dependencies 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

#Load financial news dataset 
df = pd.read_csv('Financial_News_Dataset.csv', encoding = "ISO-8859-1")
df.columns = ['sentiment', 'news']

#Dataset Preprocessing 
df['news'] = df['news'].apply(str.lower)
df['news'] = df['news'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]',"",x))
tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(df['news'].values)
X = tokenizer.texts_to_sequences(df['news'].values)
X = pad_sequences(X)
y = pd.get_dummies(df['sentiment']).values

#Train Test Split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Build LSTM Model 
model = Sequential()
model.add(Embedding(5000, 256, input_length = X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout = 0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout = 0.3, recurrent_dropout =0.2))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Training the model with dataset
model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose=2)
model.save(r'SentimentModel.h5')

#Testing the Model 
model = load_model(r'SentimentModel.h5')
predictions = model.predict(X_test)
print(predictions)