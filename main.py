import os
import numpy as np
from flask import Flask, render_template, jsonify
from flask import request
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import string


app = Flask(__name__)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def load_tf_model():
    global model
    LOCATE_PY_DIRECTION_PATH = os.path.abspath(os.path.dirname(__file__))
    model = load_model(LOCATE_PY_DIRECTION_PATH + '/model3.h5')
    global graph
    graph = tf.compat.v1.get_default_graph()


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    df_origin = pd.read_csv(file, sep = ",")
    df = df_origin.copy()
    df["comment"] = df['comment'].apply(remove_punctuations)
    df["comment"] = df["comment"].apply(lambda x: x.replace("\r", " "))
    df["comment"] = df["comment"].apply(lambda x: x.replace("\n", " "))
    df['comment'] = df['comment'].str.replace('\d+', '')
    df["comment"] = df["comment"].str.lower()
    stopwords = open('turkce-stop-words.txt', 'r', encoding='utf-8').read().split()
    df['comment_clean'] = df['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    data = df['comment_clean'].values.tolist()
    df['comment'] = df_origin['comment']
    num_words = 10000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data)
    comments_tokens = tokenizer.texts_to_sequences(data)
    comments_pads = pad_sequences(comments_tokens, maxlen=150, dtype='int32', value=0)
    y_pred = model.predict(comments_pads)
    y_pred = y_pred.T[0]
    array_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
    df['point'] = array_pred
    df.to_csv("pred_tweets.csv", index = False)

    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_tf_model()
    app.run(host="0.0.0.0", port=80)