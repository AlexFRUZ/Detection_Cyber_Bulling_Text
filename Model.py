import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv(r'D:\Train\Cyber_Bulling\cyberbullying_tweets1.csv')

data['tweet_text'] = data['tweet_text'].astype(str)

valid_classes = ['religion', 'age', 'gender', 'ethnicity', 'not_cyberbullying']
data = data[data['cyberbullying_type'].isin(valid_classes)]

vocab_size = 1000
embedding_dim = 64
max_length = 50
num_classes = 5

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data['tweet_text'])
sequences = tokenizer.texts_to_sequences(data['tweet_text'])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

label_mapping = {
    'religion': 0,
    'age': 1,
    'gender': 2,
    'ethnicity': 3,
    'not_cyberbullying': 4
}
labels = data['cyberbullying_type'].map(label_mapping)
labels = to_categorical(labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy[1]}')

model.save('cyberbullying_classifier_model1.h5')
with open('tokenizer.pickle', 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
