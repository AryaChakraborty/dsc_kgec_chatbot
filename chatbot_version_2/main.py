import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import numpy as np
import random
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
import pymongo as pym
from flask import Flask, request
from keras.models import load_model
model = load_model('chatbot_model.h5')
model_two = load_model('chatbot_model_second.h5')
import pickle
classes = pickle.load(open("classes.pkl", "rb"))
words = pickle.load(open("words.pkl", "rb"))
context_classes = pickle.load(open("context_classes.pkl", "rb"))
# words_two = pickle.load(open("words_two.pkl", "rb"))

app = Flask(__name__)
url = "mongodb+srv://aryatito:Tito%40420@cluster0.5viwfw1.mongodb.net/?retryWrites=true&w=majority"

def clean_up_sentence(sentence):
# tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
# stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
# tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
# bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
# assign 1 if current word is in the vocabulary position
                bag[i] = 1
            if show_details:
                print ("found in bag: %s" % w)
    return(np.array(bag))

# function to get the context of a particular text
def predict_context(text, model) :
    p = bow(text, words, show_details=False)
    print(p)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"context": context_classes[r[0]], "probability": str(r[1])})
    return return_list[0]["context"]

def predict_class(sentence, model):
# filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    print(p)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
# sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    result = ''
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def store_context(userID="123", context="DSC") :
    client = pym.MongoClient(url) # client
    db = client["chatbot_database"] # database
    collection = db["first_collection"] # collection
    db_dict = {"chatID" : userID, "context" : context}
    if collection.count_documents({"chatID": userID}) > 0 :
        collection.update_one({"chatID": userID}, {"$set": {"context": context}})
    else :
        collection.insert_one(db_dict)

def get_context(userID="123") :
    client = pym.MongoClient(url)  # client
    db = client["chatbot_database"]  # database
    collection = db["first_collection"]  # collection
    found = collection.find_one({"chatID" : userID})
    return found["context"]

def chatbot_response(text, uid):
    text = str(text)
    # if there is an it, we replace it with the original context of the text
    if ((text.lower()).find("it") != -1) :
        context = str(get_context(userID=uid))
        text = text.replace("it", context)
    else :
        contx = predict_context(text, model)  # need to train model first
        store_context(userID=uid, context=contx)
    newtext = text
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return (res, newtext)

@app.route('/reply', methods=['POST'])
def reply_to_text():
    if request.method == 'POST':
        data = request.json
        text = data["text"]
        uid = data["user_id"]
        # print(text)

        value_to_be_returned = {
            "error": False,
            "reply": chatbot_response(text, uid)[0],
            "text" : chatbot_response(text, uid)[1]
        }
    else:
        value_to_be_returned = {
            "error": True,
            "message": "Only Post Allowed"
        }
    return value_to_be_returned


if __name__ == '__main__':
    app.run(debug=True)