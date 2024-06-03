import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from fastapi import FastAPI
from googletrans import Translator

nltk.download("punkt")
nltk.download("wordnet")

app = FastAPI()
translator = Translator()

data_file = open('C:\Ghiraschatbot\intents (1).json').read()
data = json.loads(data_file)

# Extracting data_X(features) and data_Y(Target)
words = []  # For Bow model/ vocabulary for patterns
classes = []  # For Bow model/ vocabulary for tags
data_X = []  # For storing each pattern
data_y = []  # For storing tag corresponding to each pattern in data_X
# Iterating over all the intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)  # tokenize each pattern
        words.extend(tokens)  # and append tokens to words
        data_X.append(pattern)  # appending pattern to data_X
        data_y.append(intent["tag"])  # appending the associated tag to each pattern
    # Adding the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# Lemmatize all the words in the vocab and convert them to lowercase
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in string.punctuation
]

# Sorting the vocab and classes in alphabetical order and taking the set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

# Text to Numbers
training = []
out_empty = [0] * len(classes)

# Creating the bag of words model
for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    # Mark the index of class that the current pattern is associated to
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1

    # Add the one hot encoded BoW and associated classes to training
    training.append([bow, output_row])

# Shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the features and target labels
train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))

# The Neural Network Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(
    loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"]
)
print(model.summary())
model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)

# Preprocessing the Input
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]  # Extracting probabilities
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)  # Sorting by values of probability in decreasing order
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])  # Contains labels(tags) for highest probability
    return return_list
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "I'm sorry, I didn't understand. Can you please rephrase or provide more information?"
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    return result

# Interacting with the chatbot
message = input("")
translated_message = translator.translate(message, src="ar", dest="en").text
intents = pred_class(translated_message, words, classes)
response = get_response(intents, data)
translated_response = translator.translate(response, src="en", dest="ar").text
print(translated_response)

# FastAPI endpoint
@app.post("/chat")
def chat_with_bot(message: str):
    translated_message = translator.translate(message, src="ar", dest="en").text
    intents = pred_class(translated_message, words, classes)
    response = get_response(intents, data)
    translated_response = translator.translate(response, src="en", dest="ar").text
    return {"response": translated_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)